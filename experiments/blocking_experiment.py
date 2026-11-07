#!/usr/bin/env python3
"""Step 2 — candidate generation (blocking) experiment. No model is trained here.

Phases
  0. featurise   stream train_source{1,2,3}.tsv into a feature store (src/ber/features.py) and check
                 that every exact-key hash equals the EDA cache (same normalisation).
  1. design      FIT fold only: a seeded sample of fit S1 queries against the full FIT pool (8.26M
                 S2+S3 records — the size of the real test pool). Measures every channel and chooses
                 the retrieval budget, per-channel K and the union with pre-stated rules.
  2. evaluate    held-out VALIDATION split (val S1 vs val pool = experiments/validation_split):
                 every channel independently + the chosen union, edge cases, singletons, misses;
                 writes candidate_pairs.tsv for the validation experiment and validates it with the
                 official validator's own checking function.
  3. stress      validation S1 vs the FULL training pool (val pool + fit pool, 10.3M records) — the
                 fit records are true non-matches for validation S1 — to emulate test-scale density.

Outputs: experiments/blocking/{BLOCKING_REPORT.md, blocking_results.json, blocking_config.json,
         val_candidate_pairs.tsv, features/}

    python experiments/blocking_experiment.py [--workers 6] [--design-queries 100000]
"""

import argparse
import contextlib
import importlib.util
import io
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from ber import blocking as B  # noqa: E402
from ber import blocking_eval as E  # noqa: E402
from ber import normalize as N  # noqa: E402
from ber.features import FeatureStore, featurise_file  # noqa: E402
from ber.validation_split import IdIndex, seeded_hash  # noqa: E402

SEED = 20260925
EXACT_SPECS = [
    ("name_n1",), ("name_n2",), ("name_n3",), ("addr_n1",), ("addr_n2",),
    ("name_n2", "country"), ("addr_n2", "country"),
    ("name_n1", "addr_n1"), ("name_n2", "addr_n2"), ("name_n3", "addr_n2"),
    ("name_n3", "house"), ("name_tok1", "house"), ("name_tok1", "addr_digits"),
    ("postal", "name_tok1"), ("postal", "house"), ("house", "addr_digits"),
]
EXACT_CAPS = [None, 1000, 100, 20]
RETRIEVAL_CHANNELS = ["name", "addr", "nameaddr"]
K_MAX = 50
KS = [1, 2, 3, 5, 10, 20, 30, 50]
DEFAULT_MAX_POSTINGS = 100_000
BUDGET_GRID = [20_000, 100_000, 500_000]      # sensitivity, nameaddr channel, design only
# union grid (design): per-source K for each retrieval channel (0 = channel off), exact set on/off
GRID_NAMEADDR = [10, 20, 30, 50]
GRID_NAME = [0, 10, 20]
GRID_ADDR = [0, 10, 20]
# pre-stated selection rules (applied to FIT-fold numbers only)
EXACT_UNION_CAP = 100
EXACT_UNION_MIN_PRECISION = 0.05     # exact channel joins the union if its fit precision at cap >= 5 %
MAX_MEAN_CANDIDATES = 100            # hard budget: mean candidates per S1 (S2+S3 together)
MIN_MARGINAL_GAIN = 0.0002           # stop growing when < 0.02 pp recall per +1 mean candidate/S1


def log(*a):
    print(*a, flush=True)


def spec_name(spec):
    return "+".join(spec)


# ------------------------------------------------------------------------------------------------
class Context:
    """Feature stores, ground truth aligned to file rows, fold assignments."""

    def __init__(self, feat_dir, split_dir, cache_dir):
        self.s = {k: FeatureStore(feat_dir, f"train_s{k}") for k in (1, 2, 3)}
        self.fold = {k: np.load(Path(split_dir) / "assignment" / f"s{k}_fold.npy") for k in (1, 2, 3)}
        gt_s1 = np.load(Path(cache_dir) / "gt__s1.npy")
        pair_row = np.load(Path(cache_dir) / "gt__pair_row.npy").astype(np.int64)
        pair_t = np.load(Path(cache_dir) / "gt__pair_t.npy")
        self.pair_src = np.load(Path(cache_dir) / "gt__pair_src.npy").astype(np.int64)
        ids1 = np.asarray(self.s[1].col("id"))
        gt_row = IdIndex(ids1).rows(gt_s1)
        self.pair_s1row = gt_row[pair_row]
        self.pair_trow = np.full(len(pair_t), -1, dtype=np.int64)
        for k in (2, 3):
            m = self.pair_src == k
            self.pair_trow[m] = IdIndex(np.asarray(self.s[k].col("id"))).rows(pair_t[m])
        assert (self.pair_s1row >= 0).all() and (self.pair_trow >= 0).all()
        self.n1 = self.s[1].n

    def truth(self, q_rows):
        return E.truth_codes(q_rows, self.n1, self.pair_s1row, self.pair_src, self.pair_trow)

    def pair_masks(self, true_codes, q_rows):
        """Edge-case masks over true pairs."""
        q, src, t = B.decode(true_codes)
        s1r = np.asarray(q_rows)[q]

        def col(store_k, name, rows):
            return np.asarray(self.s[store_k].col(name))[rows]

        def tcol(name):
            out = np.empty(len(t), dtype=np.int64)
            for k in (2, 3):
                m = src == k
                out[m] = col(k, name, t[m])
            return out
        f1 = col(1, "flags", s1r).astype(np.int64)
        ft = tcol("flags")
        k1 = {c: col(1, c, s1r) for c in ("name_raw", "name_n1", "name_n2", "name_n3", "addr_n1",
                                            "addr_n2", "postal", "house")}
        kt = {c: tcol(c) for c in k1}
        true_per_q = np.bincount(q)
        return {
            "s1_name_missing": (f1 & N.FLAG_NAME_MISSING) > 0,
            "target_name_missing": (ft & N.FLAG_NAME_MISSING) > 0,
            "target_address_missing": (ft & N.FLAG_ADDR_MISSING) > 0,
            "s1_or_target_country_missing": ((f1 | ft) & N.FLAG_COUNTRY_MISSING) > 0,
            "target_name_non_latin": (ft & N.FLAG_NAME_NONLATIN) > 0,
            "target_name_domain_like": (ft & N.FLAG_NAME_DOMAIN) > 0,
            "target_name_junk_prefix": (ft & N.FLAG_NAME_JUNK_PREFIX) > 0,
            "address_landmark_either_side": ((f1 | ft) & N.FLAG_ADDR_LANDMARK) > 0,
            "name_punctuation_or_casing_only": (k1["name_raw"] != kt["name_raw"]) & (k1["name_n1"] == kt["name_n1"]),
            "name_legal_suffix_only": (k1["name_n1"] != kt["name_n1"]) & (k1["name_n2"] == kt["name_n2"]),
            "name_word_order_only": (k1["name_n2"] != kt["name_n2"]) & (k1["name_n3"] == kt["name_n3"]),
            "name_other_difference": k1["name_n3"] != kt["name_n3"],
            "address_reordered_or_abbreviated": (k1["addr_n1"] != kt["addr_n1"]) & (k1["addr_n2"] == kt["addr_n2"]) & (kt["addr_n2"] != 0),
            "postal_present_one_side_only": (k1["postal"] == 0) != (kt["postal"] == 0),
            "postal_both_present_differ": (k1["postal"] != 0) & (kt["postal"] != 0) & (k1["postal"] != kt["postal"]),
            "house_number_differs": (k1["house"] != 0) & (kt["house"] != 0) & (k1["house"] != kt["house"]),
            "s1_with_multiple_true_matches": true_per_q[q] >= 2,
        }


def same_country_space(ctx, q_rows, pool_rows):
    qc = np.asarray(ctx.s[1].col("country"))[q_rows]
    u, c = np.unique(qc, return_counts=True)
    tot = 0
    for k in (2, 3):
        pc = np.asarray(ctx.s[k].col("country"))[pool_rows[k]]
        pu, pcnt = np.unique(pc, return_counts=True)
        d = dict(zip(pu.tolist(), pcnt.tolist()))
        tot += sum(int(ci) * d.get(int(ui), 0) for ui, ci in zip(u, c))
    return tot


# ------------------------------------------------------------------------------------------------
def measure_exact(ctx, q_rows, pool_rows, true_codes, caps=EXACT_CAPS):
    """Exact channels measured from block sizes only (no pair materialisation)."""
    nq = len(q_rows)
    tq, tsrc, tt = B.decode(true_codes)
    cart = nq * (len(pool_rows[2]) + len(pool_rows[3]))
    out = {}
    for spec in EXACT_SPECS:
        qk = B.key_array(ctx.s[1], spec, q_rows)
        per_cap_counts = {str(c): np.zeros(nq, dtype=np.int64) for c in caps}
        hit = {str(c): np.zeros(len(true_codes), dtype=bool) for c in caps}
        for k in (2, 3):
            pr = pool_rows[k]
            pk = B.key_array(ctx.s[k], spec, pr)
            _, cnt, _ = B.exact_block_sizes(qk, pk)
            m = tsrc == k
            # key of each true target (file row -> key)
            tkey = B.key_array(ctx.s[k], spec, tt[m])
            eq = (tkey == qk[tq[m]]) & (tkey != 0)
            for c in caps:
                cc = cnt if c is None else np.where(cnt > c, 0, cnt)
                per_cap_counts[str(c)] += cc
                hit[str(c)][np.flatnonzero(m)] = eq & (cc[tq[m]] > 0)
        res = {}
        for c in caps:
            cnts = per_cap_counts[str(c)]
            n_c = int(cnts.sum())
            h = hit[str(c)]
            res["none" if c is None else str(c)] = {
                "candidate_recall": float(h.mean()), "true_pairs_recovered": int(h.sum()),
                "recall_s2": float(h[tsrc == 2].mean()), "recall_s3": float(h[tsrc == 3].mean()),
                "n_candidate_pairs": n_c, "precision": int(h.sum()) / n_c if n_c else 0.0,
                "candidates_per_s1": E._stats(cnts), "reduction_ratio": 1 - n_c / cart}
        out[spec_name(spec)] = res
        log(f"    exact {spec_name(spec):24s} recall(none)={res['none']['candidate_recall']:.4f} "
            f"recall(cap100)={res['100']['candidate_recall']:.4f} pairs(cap100)={res['100']['n_candidate_pairs']:,}")
    return out


def run_retrieval(ctx, q_rows, pool_rows, true_codes, channel, k, max_postings):
    codes, ranks = [], []
    t0 = time.time()
    for src in (2, 3):
        c, r, _ = B.retrieval_channel(ctx.s[1], q_rows, ctx.s[src], pool_rows[src], src, channel, k,
                                      max_postings)
        codes.append(c)
        ranks.append(r)
    codes, ranks = np.concatenate(codes), np.concatenate(ranks)
    rk, true_rank = E.recall_at_k(codes, ranks, true_codes, len(q_rows), KS)
    return codes, ranks, rk, true_rank, time.time() - t0


def materialise_exact(ctx, q_rows, pool_rows, specs, cap):
    out = []
    for spec in specs:
        qk = B.key_array(ctx.s[1], spec, q_rows)
        for k in (2, 3):
            pk = B.key_array(ctx.s[k], spec, pool_rows[k])
            out.append(B.exact_channel(qk, pk, pool_rows[k], k, cap))
    return np.unique(np.concatenate(out)) if out else np.empty(0, np.int64)


def union_metrics(parts, true_codes, nq):
    u = np.unique(np.concatenate(parts))
    hit = np.isin(true_codes, u, assume_unique=True)
    return float(hit.mean()), len(u) / nq, len(u)


# ------------------------------------------------------------------------------------------------
def phase_featurise(args, R):
    log("[0] featurise train files")
    t0 = time.time()
    metas = {}
    for k in (1, 2, 3):
        metas[f"train_s{k}"] = featurise_file(Path(args.data_dir) / f"train_source{k}.tsv", args.feat_dir,
                                              f"train_s{k}", f"S{k}-", workers=args.workers, log=log)
    # consistency with the EDA cache
    check = {}
    for k in (1, 2, 3):
        st = FeatureStore(args.feat_dir, f"train_s{k}")
        cols = ["id"] + N.KEY_COLS
        check[f"train_s{k}"] = {c: bool(np.array_equal(np.asarray(st.col(c)),
                                                       np.load(Path(args.cache_dir) / f"train_s{k}__{c}.npy")))
                                for c in cols}
    ok = all(all(v.values()) for v in check.values())
    log(f"  exact-key hashes identical to EDA cache: {ok}")
    R["featurise"] = {"meta": metas, "identical_to_eda_cache": ok, "per_column": check,
                      "seconds": round(time.time() - t0, 1)}
    if not ok:
        raise RuntimeError("feature keys differ from the EDA cache — normalisation drift")


def phase_design(ctx, args, R):
    log("[1] design on FIT fold")
    fit_s1 = np.flatnonzero(ctx.fold[1] == 0)
    keys = seeded_hash([f"S1-{int(x)}" for x in np.asarray(ctx.s[1].col("id"))[fit_s1]], SEED)
    q_rows = np.sort(fit_s1[np.argsort(keys, kind="stable")[:args.design_queries]])
    pool_rows = {k: np.flatnonzero(ctx.fold[k] == 0) for k in (2, 3)}
    true = ctx.truth(q_rows)
    nq = len(q_rows)
    D = {"n_queries": nq, "n_true_pairs": int(len(true)),
         "pool_size": {k: int(len(v)) for k, v in pool_rows.items()}}
    log(f"  queries={nq:,} true pairs={len(true):,} pool S2={len(pool_rows[2]):,} S3={len(pool_rows[3]):,}")

    D["exact"] = measure_exact(ctx, q_rows, pool_rows, true)

    # budget sensitivity on the combined channel
    D["budget_sensitivity_nameaddr"] = {}
    for mp in BUDGET_GRID:
        _, _, rk, _, secs = run_retrieval(ctx, q_rows, pool_rows, true, "nameaddr", K_MAX, mp)
        D["budget_sensitivity_nameaddr"][str(mp)] = {"recall_at_k": rk, "seconds": round(secs, 1)}
        log(f"    nameaddr budget={mp:,}: R@10={rk['10']['candidate_recall']:.4f} "
            f"R@50={rk['50']['candidate_recall']:.4f} ({secs:.0f}s)")
    # budget rule: smallest budget whose R@50 is within 0.1 pp of the largest budget's R@50
    best = D["budget_sensitivity_nameaddr"][str(BUDGET_GRID[-1])]["recall_at_k"]["50"]["candidate_recall"]
    max_postings = next(mp for mp in BUDGET_GRID
                        if D["budget_sensitivity_nameaddr"][str(mp)]["recall_at_k"]["50"]["candidate_recall"]
                        >= best - 0.001)
    D["chosen_max_postings"] = max_postings
    log(f"  chosen max_postings={max_postings:,}")

    runs = {}
    D["retrieval"] = {}
    for ch in RETRIEVAL_CHANNELS:
        codes, ranks, rk, _, secs = run_retrieval(ctx, q_rows, pool_rows, true, ch, K_MAX, max_postings)
        runs[ch] = (codes, ranks)
        D["retrieval"][ch] = {"recall_at_k": rk, "seconds": round(secs, 1)}
        log(f"    retrieval {ch:9s} R@5={rk['5']['candidate_recall']:.4f} R@10={rk['10']['candidate_recall']:.4f} "
            f"R@20={rk['20']['candidate_recall']:.4f} R@50={rk['50']['candidate_recall']:.4f} ({secs:.0f}s)")

    # exact channels admitted to the union (rule stated above)
    exact_union = [spec for spec in EXACT_SPECS
                   if D["exact"][spec_name(spec)][str(EXACT_UNION_CAP)]["precision"] >= EXACT_UNION_MIN_PRECISION]
    D["exact_union_specs"] = [spec_name(s) for s in exact_union]
    exact_codes = materialise_exact(ctx, q_rows, pool_rows, exact_union, EXACT_UNION_CAP)
    r_ex, m_ex, _ = union_metrics([exact_codes], true, nq)
    D["exact_union_alone"] = {"candidate_recall": r_ex, "mean_candidates_per_s1": m_ex}
    log(f"  exact union ({len(exact_union)} keys, cap {EXACT_UNION_CAP}): recall={r_ex:.4f} mean={m_ex:.1f}")

    grid = []
    for kna in GRID_NAMEADDR:
        for kn in GRID_NAME:
            for ka in GRID_ADDR:
                for use_ex in (False, True):
                    parts = [runs["nameaddr"][0][runs["nameaddr"][1] < kna]]
                    if kn:
                        parts.append(runs["name"][0][runs["name"][1] < kn])
                    if ka:
                        parts.append(runs["addr"][0][runs["addr"][1] < ka])
                    if use_ex:
                        parts.append(exact_codes)
                    rec, mean_c, n_c = union_metrics(parts, true, nq)
                    grid.append({"k_nameaddr": kna, "k_name": kn, "k_addr": ka, "exact": use_ex,
                                 "candidate_recall": rec, "mean_candidates_per_s1": mean_c})
    D["union_grid"] = grid
    # frontier + pre-stated selection rule
    pts = sorted(grid, key=lambda g: (g["mean_candidates_per_s1"], -g["candidate_recall"]))
    frontier = []
    for g in pts:
        if not frontier or g["candidate_recall"] > frontier[-1]["candidate_recall"]:
            frontier.append(g)
    D["union_frontier"] = frontier
    feasible = [g for g in frontier if g["mean_candidates_per_s1"] <= MAX_MEAN_CANDIDATES]
    chosen = feasible[0]
    for g in feasible[1:]:
        gain = (g["candidate_recall"] - chosen["candidate_recall"]) / \
               max(g["mean_candidates_per_s1"] - chosen["mean_candidates_per_s1"], 1e-9)
        if gain < MIN_MARGINAL_GAIN:
            break
        chosen = g
    D["chosen_union"] = chosen
    log(f"  chosen union: {chosen}")
    config = {
        "description": "Blocking configuration chosen on the FIT fold (experiments/blocking_experiment.py)",
        "exact": [{"key": list(s), "cap": EXACT_UNION_CAP} for s in exact_union] if chosen["exact"] else [],
        "retrieval": [{"channel": "nameaddr", "k": chosen["k_nameaddr"], "max_postings": max_postings}]
        + ([{"channel": "name", "k": chosen["k_name"], "max_postings": max_postings}] if chosen["k_name"] else [])
        + ([{"channel": "addr", "k": chosen["k_addr"], "max_postings": max_postings}] if chosen["k_addr"] else []),
        "country_partitioned_retrieval": True,
        "per_source_k": True,
    }
    R["design"] = D
    return config


def phase_evaluate(ctx, args, R, config, label, q_rows, pool_rows, write_tsv=None, full_channels=True):
    log(f"[{label}] queries={len(q_rows):,} pool S2={len(pool_rows[2]):,} S3={len(pool_rows[3]):,}")
    true = ctx.truth(q_rows)
    nq = len(q_rows)
    pool_size = {k: len(v) for k, v in pool_rows.items()}
    V = {"n_queries": nq, "n_true_pairs": int(len(true)), "pool_size": pool_size}
    masks = ctx.pair_masks(true, q_rows)
    scs = same_country_space(ctx, q_rows, pool_rows)
    mp = config["retrieval"][0]["max_postings"]
    per_channel = {}
    t0 = time.time()
    if full_channels:
        V["exact"] = measure_exact(ctx, q_rows, pool_rows, true)
        V["retrieval"] = {}
        for ch in RETRIEVAL_CHANNELS:
            codes, ranks, rk, true_rank, secs = run_retrieval(ctx, q_rows, pool_rows, true, ch, K_MAX, mp)
            V["retrieval"][ch] = {"recall_at_k": rk, "seconds": round(secs, 1)}
            log(f"    retrieval {ch:9s} R@10={rk['10']['candidate_recall']:.4f} R@50={rk['50']['candidate_recall']:.4f}")
            kk = next((r["k"] for r in config["retrieval"] if r["channel"] == ch), None)
            if kk:
                # top-k is exactly the rank<k prefix of the deterministic top-50 (heap order: score, doc)
                sel = np.unique(codes[ranks < kk])
                per_channel[f"retrieval:{ch}@{kk}"] = sel
                V["retrieval"][ch][f"at_chosen_k_{kk}"], _ = E.evaluate(sel, true, nq, pool_size, scs, masks)
            del codes, ranks
        for ex in config.get("exact", []):
            per_channel["exact:" + "+".join(ex["key"])] = materialise_exact(
                ctx, q_rows, pool_rows, [tuple(ex["key"])], ex.get("cap"))
        union = np.unique(np.concatenate(list(per_channel.values())))
    else:
        union, per_channel = B.generate_candidates(config, ctx.s[1], q_rows, {2: ctx.s[2], 3: ctx.s[3]},
                                                   pool_rows, log=log)
    V["union_seconds"] = round(time.time() - t0, 1)
    V["union"], hit = E.evaluate(union, true, nq, pool_size, scs, masks)
    # per-channel contribution inside the union
    contrib = {}
    covered = {n: np.isin(true, c, assume_unique=True) for n, c in per_channel.items()}
    for n, cov in covered.items():
        others = np.zeros(len(true), dtype=bool)
        for n2, c2 in covered.items():
            if n2 != n:
                others |= c2
        contrib[n] = {"candidate_recall_alone": float(cov.mean()), "n_pairs": int(len(per_channel[n])),
                      "unique_true_pairs": int((cov & ~others).sum())}
    V["union_channel_contribution"] = contrib
    log(f"  UNION recall={V['union']['candidate_recall']:.4f} pairs={V['union']['n_candidate_pairs']:,} "
        f"mean/S1={V['union']['candidates_per_s1']['mean']:.1f} RR={V['union']['reduction_ratio']:.6f}")
    if write_tsv:
        ids = {k: np.asarray(ctx.s[k].col("id")) for k in (1, 2, 3)}
        q_ids = [f"S1-{int(x)}" for x in ids[1][q_rows]]
        B.write_candidate_pairs(write_tsv, union, q_ids, {2: ids[2], 3: ids[3]})
        V["candidate_pairs_tsv"] = str(write_tsv)
    V["_missed"] = true[~hit]
    V["_missed_breakdown"] = {k: int((m & ~hit).sum()) for k, m in masks.items()}
    return V


def official_check_candidates(path, split_dir):
    spec = importlib.util.spec_from_file_location("official_validator", ROOT / "utils" / "validate_submission.py")
    official = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(official)
    tdir = Path(split_dir) / "val_as_test_dir"
    required = official.read_ids(str(tdir / "test_source1.tsv"))
    valid = official.read_ids(str(tdir / "test_source2.tsv")) | official.read_ids(str(tdir / "test_source3.tsv"))
    errors = []
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        official.validate_id_list_file(str(path), official.CANDIDATE_HEADER, "candidate_entity_ids",
                                       required, valid, errors)
    return {"errors": errors, "stdout": buf.getvalue().strip()}


def missed_examples(ctx, args, missed_codes, q_rows, n=30):
    import eda  # experiments/eda.py — pair_tags for describing differences
    rng = np.random.default_rng(SEED)
    pick = np.sort(rng.choice(len(missed_codes), size=min(n, len(missed_codes)), replace=False))
    q, src, t = B.decode(missed_codes[pick])
    s1r = np.asarray(q_rows)[q]
    need = {1: set(s1r.tolist()), 2: set(t[src == 2].tolist()), 3: set(t[src == 3].tolist())}
    lines = {}
    for k in (1, 2, 3):
        want, got = need[k], {}
        if want:
            with open(Path(args.data_dir) / f"train_source{k}.tsv", "rb") as f:
                f.readline()
                row = 0
                last = max(want)
                for b in f:
                    if not b.strip():
                        continue
                    if row in want:
                        got[row] = (b.decode("utf-8").rstrip("\r\n").split("\t") + [""] * 4)[:4]
                    if row >= last:
                        break
                    row += 1
        lines[k] = got
    out = []
    tag_counts = {}
    for a, s, b in zip(s1r, src, t):
        r1, r2 = lines[1][int(a)], lines[int(s)][int(b)]
        tags, _, _ = eda.pair_tags(r1, r2)
        for tg in tags:
            tag_counts[tg] = tag_counts.get(tg, 0) + 1
        out.append({"s1": r1, "target": r2, "differences": tags})
    return out, tag_counts


# ------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------
# Checkpoints (one JSON per completed phase, written atomically as soon as the phase finishes)
# ------------------------------------------------------------------------------------------------
MIN_FREE_BYTES = 1_000_000_000      # stress guard: checkpoint + stop cleanly below 1 GB free RAM


def ckpt_path(out, name):
    return Path(out) / "checkpoints" / f"{name}.json"


def save_ckpt(out, name, obj):
    p = ckpt_path(out, name)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(to_jsonable(obj), indent=1, ensure_ascii=False), encoding="utf-8")
    tmp.replace(p)
    log(f"  checkpoint saved: {p}")


def load_ckpt(out, name):
    p = ckpt_path(out, name)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def free_bytes():
    try:
        import psutil
        return psutil.virtual_memory().available
    except Exception:
        return None


class MemoryUnsafe(RuntimeError):
    pass


def guard(stage):
    fb = free_bytes()
    if fb is not None and fb < MIN_FREE_BYTES:
        raise MemoryUnsafe(f"free RAM {fb / 1e9:.2f} GB < {MIN_FREE_BYTES / 1e9:.1f} GB before {stage}")


# ------------------------------------------------------------------------------------------------
# Recovery of the completed design/validation numbers from the preserved run log
# ------------------------------------------------------------------------------------------------
_F = r"([0-9.]+)"
_N = r"([0-9,]+)"


def recover_from_log(log_path, config):
    """Parse the exact figures printed by a completed design + validation run.

    Only lines emitted by this script are parsed; values are exactly those printed (4 decimals for
    recalls). Consistency with the saved blocking_config.json is checked."""
    import ast
    import re
    text = Path(log_path).read_text(encoding="utf-8")
    if "[1] design on FIT fold" not in text or "[2 evaluate]" not in text:
        raise ValueError(f"{log_path} does not contain a completed design + evaluate run")
    design_txt = text.split("[1] design on FIT fold", 1)[1].split("[2 evaluate]", 1)[0]
    eval_txt = text.split("[2 evaluate]", 1)[1].split("[3 stress]", 1)[0]
    num = lambda x: int(x.replace(",", ""))  # noqa: E731

    def exact_lines(t):
        out = {}
        for m in re.finditer(rf"exact (\S+)\s+recall\(none\)={_F} recall\(cap100\)={_F} pairs\(cap100\)={_N}", t):
            out[m.group(1)] = {"recall_uncapped": float(m.group(2)), "recall_cap100": float(m.group(3)),
                               "pairs_cap100": num(m.group(4))}
        return out

    D = {"source": "recovered from run log (design completed; full JSON was not yet checkpointed)"}
    m = re.search(rf"queries={_N} true pairs={_N} pool S2={_N} S3={_N}", design_txt)
    D["n_queries"], D["n_true_pairs"] = num(m.group(1)), num(m.group(2))
    D["pool_size"] = {"2": num(m.group(3)), "3": num(m.group(4))}
    D["exact"] = exact_lines(design_txt)
    D["budget_sensitivity_nameaddr"] = {
        m.group(1).replace(",", ""): {"R@10": float(m.group(2)), "R@50": float(m.group(3)), "seconds": float(m.group(4))}
        for m in re.finditer(rf"nameaddr budget={_N}: R@10={_F} R@50={_F} \({_F}s\)", design_txt)}
    D["chosen_max_postings"] = num(re.search(rf"chosen max_postings={_N}", design_txt).group(1))
    D["retrieval"] = {
        m.group(1): {"R@5": float(m.group(2)), "R@10": float(m.group(3)), "R@20": float(m.group(4)),
                     "R@50": float(m.group(5)), "seconds": float(m.group(6))}
        for m in re.finditer(rf"retrieval (\w+)\s+R@5={_F} R@10={_F} R@20={_F} R@50={_F} \({_F}s\)", design_txt)}
    m = re.search(rf"exact union \((\d+) keys, cap (\d+)\): recall={_F} mean={_F}", design_txt)
    D["exact_union_alone"] = {"n_keys": int(m.group(1)), "cap": int(m.group(2)),
                              "candidate_recall": float(m.group(3)), "mean_candidates_per_s1": float(m.group(4))}
    D["chosen_union"] = ast.literal_eval(re.search(r"chosen union: (\{.*\})", design_txt).group(1))
    cu = D["chosen_union"]
    ks = {r["channel"]: r["k"] for r in config["retrieval"]}
    consistent = (ks.get("nameaddr") == cu["k_nameaddr"] and ks.get("name", 0) == cu["k_name"]
                  and ks.get("addr", 0) == cu["k_addr"] and bool(config["exact"]) == cu["exact"]
                  and all(r["max_postings"] == D["chosen_max_postings"] for r in config["retrieval"])
                  and (not cu["exact"] or len(config["exact"]) == D["exact_union_alone"]["n_keys"]))
    if not consistent:
        raise ValueError("recovered design does not match blocking_config.json")
    D["config_consistent_with_saved_blocking_config"] = True

    V = {"source": "per-channel figures recovered from run log (validation completed)"}
    m = re.search(rf"queries={_N} pool S2={_N} S3={_N}", text.split("[2 evaluate]", 1)[1])
    V["n_queries"], V["pool_size"] = num(m.group(1)), {"2": num(m.group(2)), "3": num(m.group(3))}
    V["exact"] = exact_lines(eval_txt)
    V["retrieval"] = {m.group(1): {"R@10": float(m.group(2)), "R@50": float(m.group(3))}
                      for m in re.finditer(rf"retrieval (\w+)\s+R@10={_F} R@50={_F}", eval_txt)}
    m = re.search(rf"UNION recall={_F} pairs={_N} mean/S1={_F} RR={_F}", eval_txt)
    V["union_logged"] = {"candidate_recall": float(m.group(1)), "n_candidate_pairs": num(m.group(2)),
                         "mean_candidates_per_s1": float(m.group(3)), "reduction_ratio": float(m.group(4))}
    V["official_candidate_check_logged"] = "official candidate check: PASS" in eval_txt
    return D, V


# ------------------------------------------------------------------------------------------------
# Validation metrics from the saved candidate_pairs.tsv (no blocking is re-run)
# ------------------------------------------------------------------------------------------------

def codes_from_candidate_tsv(ctx, tsv, q_rows):
    """Parse candidate_pairs.tsv (row order = q_rows) into sorted pair codes."""
    from array import array
    ids1 = np.asarray(ctx.s[1].col("id"))[q_rows]
    qpos, src, num = array("q"), array("b"), array("q")
    with open(tsv, encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
        if header != ["source1_entity_id", "candidate_entity_ids"]:
            raise ValueError(f"unexpected header {header}")
        for i, line in enumerate(f):
            s1, _, rest = line.partition("\t")
            if int(s1[3:]) != int(ids1[i]):
                raise ValueError(f"row {i}: {s1} is not query {i} (S1-{ids1[i]})")
            rest = rest.rstrip("\n")
            if not rest:
                continue
            for x in rest.split(","):
                qpos.append(i)
                src.append(int(x[1]))
                num.append(int(x[3:]))
    if i + 1 != len(q_rows):
        raise ValueError(f"{tsv}: {i + 1} rows for {len(q_rows)} queries")
    qpos, src, num = (np.frombuffer(a, dtype=d) for a, d in ((qpos, np.int64), (src, np.int8), (num, np.int64)))
    t = np.empty(len(num), dtype=np.int64)
    for k in (2, 3):
        m = src == k
        r = IdIndex(np.asarray(ctx.s[k].col("id"))).rows(num[m])
        if (r < 0).any():
            raise ValueError(f"{int((r < 0).sum())} S{k} candidate IDs not found in train_source{k}")
        t[m] = r
    return np.unique(B.encode(qpos, src, t))


def phase_validation_from_tsv(ctx, args, config, tsv):
    q_rows = np.flatnonzero(ctx.fold[1] == 1)
    pool_rows = {k: np.flatnonzero(ctx.fold[k] == 1) for k in (2, 3)}
    log(f"[2 evaluate-from-tsv] {tsv}")
    t0 = time.time()
    union = codes_from_candidate_tsv(ctx, tsv, q_rows)
    # every candidate must be in the validation pool (fold 1)
    _, csrc, ct = B.decode(union)
    in_pool = np.ones(len(union), dtype=bool)
    for k in (2, 3):
        m = csrc == k
        in_pool[m] = ctx.fold[k][ct[m]] == 1
    true = ctx.truth(q_rows)
    nq = len(q_rows)
    masks = ctx.pair_masks(true, q_rows)
    V = {"n_queries": nq, "n_true_pairs": int(len(true)),
         "pool_size": {k: len(v) for k, v in pool_rows.items()},
         "candidates_outside_validation_pool": int((~in_pool).sum())}
    V["union"], hit = E.evaluate(union, true, nq, V["pool_size"], same_country_space(ctx, q_rows, pool_rows), masks)
    V["missed_breakdown"] = {k: int((m & ~hit).sum()) for k, m in masks.items()}
    V["missed_examples"], V["missed_tag_counts"] = missed_examples(ctx, args, true[~hit], q_rows, n=40)
    V["seconds"] = round(time.time() - t0, 1)
    log(f"  UNION (from TSV) recall={V['union']['candidate_recall']:.4f} pairs={V['union']['n_candidate_pairs']:,} "
        f"mean/S1={V['union']['candidates_per_s1']['mean']:.1f}")
    return V


# ------------------------------------------------------------------------------------------------
# Memory-lean union (identical result to ber.blocking.generate_candidates)
# ------------------------------------------------------------------------------------------------

def lean_union(ctx, config, q_rows, pool_rows, true, partial_cb=None):
    """Channels are generated one at a time; their coverage of the true pairs is recorded and their
    codes merged into the running union, then freed. Returns (union, per-channel coverage info)."""
    union = np.empty(0, dtype=np.int64)
    cover, info = {}, {}
    jobs = [("exact:" + "+".join(ex["key"]), "exact", ex) for ex in config.get("exact", [])] + \
           [(f"retrieval:{rt['channel']}@{rt['k']}", "retrieval", rt) for rt in config.get("retrieval", [])]
    for name, kind, spec in jobs:
        guard(name)
        t0 = time.time()
        parts = []
        for src in (2, 3):
            if kind == "exact":
                qk = B.key_array(ctx.s[1], tuple(spec["key"]), q_rows)
                pk = B.key_array(ctx.s[src], tuple(spec["key"]), pool_rows[src])
                parts.append(B.exact_channel(qk, pk, pool_rows[src], src, spec.get("cap")))
                del qk, pk
            else:
                c, _, _ = B.retrieval_channel(ctx.s[1], q_rows, ctx.s[src], pool_rows[src], src,
                                              spec["channel"], spec["k"], spec["max_postings"])
                parts.append(c)
        codes = np.unique(np.concatenate(parts))
        del parts
        cover[name] = np.isin(true, codes, assume_unique=True)
        info[name] = {"n_pairs": int(len(codes)), "candidate_recall_alone": float(cover[name].mean()),
                      "seconds": round(time.time() - t0, 1)}
        union = np.union1d(union, codes)
        del codes
        log(f"    {name}: {info[name]['n_pairs']:,} pairs, recall alone {info[name]['candidate_recall_alone']:.4f}, "
            f"union so far {len(union):,} ({info[name]['seconds']}s)")
        if partial_cb:
            partial_cb(info)
    for n in info:
        others = np.zeros(len(true), dtype=bool)
        for n2, c2 in cover.items():
            if n2 != n:
                others |= c2
        info[n]["unique_true_pairs"] = int((cover[n] & ~others).sum())
    return union, info


def phase_stress(ctx, args, config):
    q_rows = np.flatnonzero(ctx.fold[1] == 1)
    pool_rows = {k: np.arange(ctx.s[k].n) for k in (2, 3)}
    log(f"[3 stress] queries={len(q_rows):,} pool S2={len(pool_rows[2]):,} S3={len(pool_rows[3]):,} "
        f"(free RAM {(free_bytes() or 0) / 1e9:.1f} GB)")
    true = ctx.truth(q_rows)
    nq = len(q_rows)
    t0 = time.time()
    union, info = lean_union(ctx, config, q_rows, pool_rows, true,
                             partial_cb=lambda inf: save_ckpt(args.out_dir, "stress_partial",
                                                              {"channels_done": inf}))
    guard("stress evaluation")
    masks = ctx.pair_masks(true, q_rows)
    S = {"n_queries": nq, "n_true_pairs": int(len(true)), "pool_size": {k: len(v) for k, v in pool_rows.items()},
         "union_seconds": round(time.time() - t0, 1), "union_channel_contribution": info}
    S["union"], hit = E.evaluate(union, true, nq, S["pool_size"], same_country_space(ctx, q_rows, pool_rows), masks)
    S["missed_breakdown"] = {k: int((m & ~hit).sum()) for k, m in masks.items()}
    log(f"  STRESS UNION recall={S['union']['candidate_recall']:.4f} pairs={S['union']['n_candidate_pairs']:,} "
        f"mean/S1={S['union']['candidates_per_s1']['mean']:.1f} RR={S['union']['reduction_ratio']:.7f}")
    return S


# ------------------------------------------------------------------------------------------------
# Report (from checkpoints; tolerant of recovered vs fully-measured phases)
# ------------------------------------------------------------------------------------------------

def fmt_pct(x, d=2):
    return "—" if x is None else f"{100 * x:.{d}f}%"


def _union_block(w, U, title):
    c = U["candidates_per_s1"]
    sc = U["candidates_per_singleton_s1"]
    w(f"### {title}\n")
    w(f"- **Candidate recall: {fmt_pct(U['candidate_recall'], 3)}** ({U['true_pairs_recovered']:,} of "
      f"{U['n_true_pairs']:,} true pairs); S2 {fmt_pct(U['recall_s2'], 3)}, S3 {fmt_pct(U['recall_s3'], 3)}")
    w(f"- Matched S1 with ALL true matches retrieved: {fmt_pct(U['matched_s1_with_all_true_recovered'])}; with at "
      f"least one: {fmt_pct(U['matched_s1_with_any_true_recovered'])}")
    w(f"- Candidate pairs: {U['n_candidate_pairs']:,} (S2 {U['n_candidate_pairs_s2']:,}, S3 {U['n_candidate_pairs_s3']:,}); "
      f"per S1 mean {c['mean']:.1f}, median {c['median']:.0f}, p90 {c['p90']:.0f}, p95 {c['p95']:.0f}, "
      f"p99 {c['p99']:.0f}, max {c['max']}; S1 with zero candidates {c['zero_candidates']:,}")
    w(f"- **Reduction ratio {U['reduction_ratio']:.7f}** vs Cartesian {U['cartesian_pairs']:,}; "
      f"{U.get('reduction_ratio_vs_same_country', float('nan')):.7f} vs same-country Cartesian "
      f"{U.get('same_country_pairs', 0):,}")
    w(f"- Zero-match S1 ({U['n_singleton_s1']:,}): candidates per singleton mean {sc['mean']:.1f}, median "
      f"{sc['median']:.0f}, p95 {sc['p95']:.0f}, max {sc['max']}; singletons with no candidate {sc['zero_candidates']:,}. "
      f"Each is a potential false merge the matcher must reject.\n")


def _rk(v, k):
    """Recall@k from either a recovered-log record ({"R@10": x}) or a measured one ({"recall_at_k": ...})."""
    return v[f"R@{k}"] if f"R@{k}" in v else v["recall_at_k"][str(k)]["candidate_recall"]


def write_report_from_ckpts(out, path):
    D, VC, VU, S = (load_ckpt(out, n) for n in ("design", "validation_channels", "validation", "stress"))
    cfg = json.loads((Path(out) / "blocking_config.json").read_text(encoding="utf-8"))
    L = []
    w = L.append
    w("# Blocking / candidate generation — Step 2 report\n")
    w("Generated by `experiments/blocking_experiment.py --phase report` from the phase checkpoints in "
      "`experiments/blocking/checkpoints/`. No model is trained. Design choices use the FIT fold only; headline "
      "numbers are on the held-out VALIDATION split.\n")
    w("**candidate_recall** = true (S1, S2/S3) pairs present in the candidate set / all true pairs (recall ceiling "
      "of any matcher, not a model score). **Reduction ratio** = 1 − candidates / (queries × pool size).\n")
    w("## Final blocking configuration (chosen on the FIT fold)\n")
    w("```json\n" + json.dumps(cfg, indent=1) + "\n```\n")
    w("Retrieval: TF-IDF cosine (binary tf, smooth idf from the pool) over hashed tokens, top-K **per source** "
      "(S2 and S3 separately), within the query's country (0 cross-country true pairs in 7.64M), rarest tokens first "
      "until `max_postings` postings are scanned. Exact channels: equality join, blocks > cap records dropped.\n")

    if VU:
        w("## Validation split (held-out) — chosen union\n")
        w(f"{VU['n_queries']:,} S1 queries; pool S2 {VU['pool_size']['2']:,} + S3 {VU['pool_size']['3']:,}. "
          f"Computed from the saved `val_candidate_pairs.tsv` (candidates outside the validation pool: "
          f"{VU['candidates_outside_validation_pool']:,}).\n")
        _union_block(w, VU["union"], "Union")
        w("Edge cases — union recall on the true pairs of each category:\n")
        w("| case | true pairs | union recall | missed |")
        w("|---|---:|---:|---:|")
        for case, v in VU["union"]["recall_by_case"].items():
            w(f"| {case} | {v['n_true_pairs']:,} | {fmt_pct(v['recall'])} | {VU['missed_breakdown'].get(case, 0):,} |")
        w("")
    if VC:
        w("## Every channel on the validation split\n")
        if VC.get("source"):
            w(f"_{VC['source']}; values as printed (4 decimals)._\n")
        w("Exact-key channels (cap = max pool records per key per source; larger blocks dropped):\n")
        w("| channel | recall (no cap) | recall (cap 100) | pairs (cap 100) | mean pairs/S1 (cap 100) | reduction ratio (cap 100) |")
        w("|---|---:|---:|---:|---:|---:|")
        cart = VC["n_queries"] * (VC["pool_size"]["2"] + VC["pool_size"]["3"])
        for n, v in VC["exact"].items():
            w(f"| `{n}` | {fmt_pct(v['recall_uncapped'])} | {fmt_pct(v['recall_cap100'])} | {v['pairs_cap100']:,} | "
              f"{v['pairs_cap100'] / VC['n_queries']:.2f} | {1 - v['pairs_cap100'] / cart:.7f} |")
        w("\nRetrieval channels (top-K per source ⇒ ≤ 2K candidates per S1; max_postings = "
          f"{cfg['retrieval'][0]['max_postings']:,}):\n")
        w("| channel | recall @10/source (≤20 per S1) | recall @50/source (≤100 per S1) |")
        w("|---|---:|---:|")
        for n, v in VC["retrieval"].items():
            w(f"| `{n}` | {fmt_pct(v['R@10'])} | {fmt_pct(v['R@50'])} |")
        u = VC["union_logged"]
        w(f"\nLogged union at run time: recall {fmt_pct(u['candidate_recall'])}, {u['n_candidate_pairs']:,} pairs, "
          f"{u['mean_candidates_per_s1']:.1f}/S1, RR {u['reduction_ratio']:.6f}; official candidate check at run time: "
          f"{'PASS' if VC['official_candidate_check_logged'] else 'not recorded'}.\n")
    if S:
        w("## Test-scale stress: validation S1 vs the FULL training pool\n")
        w(f"Pool S2 {S['pool_size']['2']:,} + S3 {S['pool_size']['3']:,} (real test pool: 4,887,273 + 5,082,316). "
          f"Same configuration; the extra 8.26M fit records are true non-matches for validation S1.\n")
        _union_block(w, S["union"], "Union")
        w("| channel | pairs | recall alone | unique true pairs | seconds |")
        w("|---|---:|---:|---:|---:|")
        for n, v in S["union_channel_contribution"].items():
            w(f"| `{n}` | {v['n_pairs']:,} | {fmt_pct(v['candidate_recall_alone'])} | {v['unique_true_pairs']:,} | {v['seconds']} |")
        w("\nEdge cases (stress):\n")
        w("| case | true pairs | union recall |")
        w("|---|---:|---:|")
        for case, v in S["union"]["recall_by_case"].items():
            w(f"| {case} | {v['n_true_pairs']:,} | {fmt_pct(v['recall'])} |")
        w("")
    if VU:
        w("## Failure cases (validation union misses)\n")
        w("Difference tags on a random sample of 40 missed pairs (seed fixed):\n")
        for k, v in sorted(VU["missed_tag_counts"].items(), key=lambda kv: -kv[1]):
            w(f"- {k}: {v}")
        w("\nRepresentative missed true matches (actual validation records):\n")
        for e in VU["missed_examples"][:25]:
            s1, t = e["s1"], e["target"]
            w(f"- **{s1[1]}** | {s1[2]} ({s1[3]})  ⟷  `{t[0]}` **{t[1]}** | {t[2] or '∅'} ({t[3]}) — "
              f"{', '.join(e['differences'])}")
        w("")
    if D:
        w("## Design (FIT fold only)\n")
        if D.get("source"):
            w(f"_{D['source']}._\n")
        w(f"Queries {D['n_queries']:,} fit S1; pool fit S2 {D['pool_size']['2']:,} + S3 {D['pool_size']['3']:,}; "
          f"true pairs {D['n_true_pairs']:,}.\n")
        w("| retrieval budget (nameaddr) | R@10 | R@50 | seconds |")
        w("|---:|---:|---:|---:|")
        for b, v in D["budget_sensitivity_nameaddr"].items():
            w(f"| {int(b):,} | {fmt_pct(_rk(v, 10))} | {fmt_pct(_rk(v, 50))} | {v['seconds']} |")
        w(f"\nChosen max_postings (smallest budget within 0.1 pp of the largest budget's R@50): "
          f"**{D['chosen_max_postings']:,}**.\n")
        w("| retrieval channel (fit) | R@5 | R@10 | R@20 | R@50 |")
        w("|---|---:|---:|---:|---:|")
        for n, v in D["retrieval"].items():
            w(f"| `{n}` | {fmt_pct(_rk(v, 5))} | {fmt_pct(_rk(v, 10))} | {fmt_pct(_rk(v, 20))} | {fmt_pct(_rk(v, 50))} |")
        eu = D["exact_union_alone"]
        n_keys = eu.get("n_keys", len(D.get("exact_union_specs", [])))
        w(f"\nExact union alone ({n_keys} keys, cap {eu.get('cap', EXACT_UNION_CAP)}): recall {fmt_pct(eu['candidate_recall'])}, "
          f"{eu['mean_candidates_per_s1']:.1f}/S1. Chosen union (grid rule: marginal gain ≥ 0.02 pp per +1 candidate/S1, "
          f"mean ≤ 100): `{D['chosen_union']}`.\n")
    Path(path).write_text("\n".join(L), encoding="utf-8")
    log(f"  report written: {path}")


def to_jsonable(o):
    if isinstance(o, dict):
        return {str(k): to_jsonable(v) for k, v in o.items() if not str(k).startswith("_")}
    if isinstance(o, (list, tuple)):
        return [to_jsonable(v) for v in o]
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return to_jsonable(o.tolist())
    return o


PHASES = ["all", "design", "evaluate", "recover-log", "validation-from-tsv", "stress", "report"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=str(ROOT / "dataset" / "train"))
    ap.add_argument("--split-dir", default=str(ROOT / "experiments" / "validation_split"))
    ap.add_argument("--cache-dir", default=str(ROOT / "experiments" / "_eda_cache"))
    ap.add_argument("--out-dir", default=str(ROOT / "experiments" / "blocking"))
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--design-queries", type=int, default=100_000)
    ap.add_argument("--skip-stress", action="store_true")
    ap.add_argument("--phase", choices=PHASES, default="all",
                    help="all: run every phase whose checkpoint is missing (completed phases are reused). "
                         "recover-log: rebuild design + per-channel validation checkpoints from a preserved run "
                         "log (--log). validation-from-tsv: validation metrics from the saved candidate TSV. "
                         "stress / report: run just that phase.")
    ap.add_argument("--log", default=None, help="run log for --phase recover-log")
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    args.out_dir = str(out)
    args.feat_dir = str(out / "features")
    cfg_path = out / "blocking_config.json"
    tsv = out / "val_candidate_pairs.tsv"
    t0 = time.time()

    def context():
        R = {}
        phase_featurise(args, R)                       # reuses the feature store when up to date
        save_ckpt(out, "featurise", R["featurise"])
        return Context(args.feat_dir, args.split_dir, args.cache_dir)

    if args.phase == "report":
        write_report_from_ckpts(out, out / "BLOCKING_REPORT.md")
        return
    if args.phase == "recover-log":
        config = json.loads(cfg_path.read_text(encoding="utf-8"))
        D, V = recover_from_log(args.log, config)
        save_ckpt(out, "design", D)
        save_ckpt(out, "validation_channels", V)
        return

    ctx = context()
    try:
        if args.phase in ("all", "design") and (args.phase == "design" or load_ckpt(out, "design") is None):
            R = {}
            config = phase_design(ctx, args, R)
            cfg_path.write_text(json.dumps(config, indent=1), encoding="utf-8")
            save_ckpt(out, "design", R["design"])
        config = json.loads(cfg_path.read_text(encoding="utf-8"))

        if args.phase in ("all", "evaluate") and (args.phase == "evaluate" or load_ckpt(out, "validation") is None):
            R = {}
            q_val = np.flatnonzero(ctx.fold[1] == 1)
            pool_val = {k: np.flatnonzero(ctx.fold[k] == 1) for k in (2, 3)}
            V = phase_evaluate(ctx, args, R, config, "2 evaluate", q_val, pool_val, write_tsv=tsv)
            V["official_candidate_check"] = official_check_candidates(tsv, args.split_dir)
            log(f"  official candidate check: {'PASS' if not V['official_candidate_check']['errors'] else 'FAIL'}")
            missed = V.pop("_missed")
            V["missed_breakdown"] = V.pop("_missed_breakdown")
            V["missed_examples"], V["missed_tag_counts"] = missed_examples(ctx, args, missed, q_val, n=40)
            V["candidates_outside_validation_pool"] = 0
            ch = {"source": None, "n_queries": V["n_queries"], "pool_size": V["pool_size"],
                  "exact": {n: {"recall_uncapped": r["none"]["candidate_recall"],
                                "recall_cap100": r["100"]["candidate_recall"],
                                "pairs_cap100": r["100"]["n_candidate_pairs"]} for n, r in V["exact"].items()},
                  "retrieval": {n: {"R@10": r["recall_at_k"]["10"]["candidate_recall"],
                                    "R@50": r["recall_at_k"]["50"]["candidate_recall"]} for n, r in V["retrieval"].items()},
                  "union_logged": {"candidate_recall": V["union"]["candidate_recall"],
                                   "n_candidate_pairs": V["union"]["n_candidate_pairs"],
                                   "mean_candidates_per_s1": V["union"]["candidates_per_s1"]["mean"],
                                   "reduction_ratio": V["union"]["reduction_ratio"]},
                  "official_candidate_check_logged": not V["official_candidate_check"]["errors"]}
            save_ckpt(out, "validation_channels", ch)
            save_ckpt(out, "validation_full", V)
            save_ckpt(out, "validation", V)

        if args.phase == "validation-from-tsv":
            V = phase_validation_from_tsv(ctx, args, config, tsv)
            save_ckpt(out, "validation", V)

        if (args.phase == "stress" or (args.phase == "all" and not args.skip_stress
                                       and load_ckpt(out, "stress") is None)):
            S = phase_stress(ctx, args, config)
            save_ckpt(out, "stress", S)
            p = ckpt_path(out, "stress_partial")
            if p.exists():
                p.unlink()
    except MemoryUnsafe as e:
        log(f"STOPPED SAFELY — {e}. Completed phases are checkpointed in {out / 'checkpoints'}.")
        sys.exit(3)

    write_report_from_ckpts(out, out / "BLOCKING_REPORT.md")
    log(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()

