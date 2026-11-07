#!/usr/bin/env python3
"""Blocking-miss feasibility analysis (READ-ONLY; nothing in production is modified; no full candidate regeneration).

Environments (validation S1 and labels only):
  V  validation pool (saved candidates experiments/matching/cand_val.npy, IDF idf_val_*)
  S  full training pool, test-sized (saved candidates experiments/test_scale/stress/cand_stress.npy, IDF idf_full_*)
For the TRUE pairs missed by the unchanged Step 2 blocking:
  * pair features (Step 3 code) and key equality -> failure modes
  * exact channels: key equality and whether the block was dropped by the cap (pool block size > cap)
  * targeted retrieval re-runs, ONLY for the S1 that have misses + a seeded random sample of 20,000 S1 (cost estimate):
      nameaddr top-50 (ranks give recovery for K=20/30/50), addr top-50, name top-50 (channel not in production),
      nameaddr top-10 with max_postings 200,000 (budget)
Outputs: experiments/diagnostics/blocking_miss_analysis/{results.json, run.log}
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from ber import blocking as B  # noqa: E402
from ber import match_features as MF  # noqa: E402
from ber import normalize as N  # noqa: E402
from ber.normalize import h64  # noqa: E402
from ber.validation_split import seeded_hash  # noqa: E402
from test_scale_check import sha256  # noqa: E402
from trank_diagnostic import PROTECTED  # noqa: E402

OUT = Path(__file__).resolve().parent
EXP = ROOT / "experiments"
C = MF.COL
VARIANTS = [("nameaddr", 50, 20_000), ("addr", 50, 20_000), ("name", 50, 20_000), ("nameaddr", 10, 200_000)]
KS = [10, 20, 30, 50]
SAMPLE = 20_000


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def hash_tree(d):
    return {str(p.relative_to(ROOT)).replace("\\", "/"): sha256(p) for p in sorted(Path(d).rglob("*")) if p.is_file()}


def main():
    t0 = time.time()
    hb = {f: sha256(ROOT / f) for f in PROTECTED}
    hbf = hash_tree(EXP / "final")
    from blocking_experiment import Context
    ctx = Context(EXP / "blocking" / "features", EXP / "validation_split", EXP / "_eda_cache")
    cfg = json.loads((EXP / "blocking" / "blocking_config.json").read_text(encoding="utf-8"))
    qv = np.flatnonzero(ctx.fold[1] == 1)
    nq = len(qv)
    cv = np.asarray(ctx.s[1].col("country"))[qv]
    true = ctx.truth(qv)
    stores = {2: ctx.s[2], 3: ctx.s[3]}
    envs = {"V": (EXP / "matching" / "cand_val.npy", {k: np.flatnonzero(ctx.fold[k] == 1) for k in (2, 3)},
                  EXP / "matching" / "idf_val_{}.npz"),
            "S": (EXP / "test_scale" / "stress" / "cand_stress.npy", {k: np.arange(ctx.s[k].n) for k in (2, 3)},
                  EXP / "test_scale" / "stress" / "idf_full_{}.npz")}
    rng_h = seeded_hash([f"S1-{int(x)}" for x in np.asarray(ctx.s[1].col("id"))[qv]], 20260925 + 77)
    sample_pos = np.sort(np.argsort(rng_h, kind="stable")[:SAMPLE])
    res = {"envs": {}}
    for env, (cp, pool, idfp) in envs.items():
        te = time.time()
        cand = np.load(cp)
        hit = np.isin(true, cand, assume_unique=True)
        miss = true[~hit]
        mq, ms, mt = B.decode(miss)
        r = {"true_pairs": int(len(true)), "missed": int(len(miss)), "candidate_recall": float(hit.mean()),
             "missed_S2": int((ms == 2).sum()), "missed_S3": int((ms == 3).sum()),
             "missed_US": int((cv[mq] == h64("us")).sum()), "missed_India": int((cv[mq] == h64("india")).sum()),
             "s1_with_a_miss": int(len(np.unique(mq)))}
        # ---- pair features for missed pairs
        idf_n, idf_a = (MF.IdfTable.load(str(idfp).format(c)) for c in ("name", "addr"))
        X = MF.pair_features(ctx.s[1], qv, stores, miss, idf_n, idf_a)
        f = lambda n: X[:, C[n]]  # noqa: E731
        tflag = np.empty(len(miss), dtype=np.int64)
        for k in (2, 3):
            m = ms == k
            tflag[m] = np.asarray(ctx.s[k].col("flags"))[mt[m]]
        nonlatin = (tflag & N.FLAG_NAME_NONLATIN) > 0
        addr_empty = (tflag & N.FLAG_ADDR_MISSING) > 0
        domain = (tflag & N.FLAG_NAME_DOMAIN) > 0
        house_diff = (f("house_both") == 1) & (f("eq_house") == 0)
        no_name_tok = f("name_inter") == 0
        r["characteristics"] = {
            "name_cos_mean": float(f("name_cos").mean()), "addr_cos_mean": float(f("addr_cos").mean()),
            "nameaddr_cos_median": float(np.median(f("nameaddr_cos"))),
            "share_no_shared_name_token": float(no_name_tok.mean()), "share_name_cos_ge_0.8": float((f("name_cos") >= 0.8).mean()),
            "share_addr_cos_lt_0.3": float((f("addr_cos") < 0.3).mean()), "share_addr_cos_ge_0.8": float((f("addr_cos") >= 0.8).mean()),
            "share_country_equal": float(f("eq_country").mean()),
            "share_postal_both_present": float(f("postal_both").mean()), "share_postal_equal": float(f("eq_postal").mean()),
            "share_house_both_present": float(f("house_both").mean()), "share_house_equal": float(f("eq_house").mean()),
            "share_house_differs": float(house_diff.mean()),
            "share_target_nonlatin": float(nonlatin.mean()), "share_target_addr_empty": float(addr_empty.mean()),
            "share_target_domain_name": float(domain.mean()),
            "share_eq_name_n2": float(f("eq_name_n2").mean()), "share_eq_name_n3": float(f("eq_name_n3").mean()),
            "share_eq_name_tok1": float(f("eq_name_tok1").mean()), "share_eq_addr_n2": float(f("eq_addr_n2").mean()),
            "share_eq_addr_digits": float(f("eq_addr_digits").mean())}
        # ---- exact channels: key equal? dropped by cap?
        ex = {}
        s1rows = qv[mq]
        for spec in cfg["exact"]:
            key = tuple(spec["key"])
            qk = B.key_array(ctx.s[1], key, s1rows)
            eq = np.zeros(len(miss), dtype=bool)
            dropped = np.zeros(len(miss), dtype=bool)
            for k in (2, 3):
                m = ms == k
                tk = B.key_array(ctx.s[k], key, mt[m])
                e = (qk[m] == tk) & (qk[m] != 0)
                pk = np.sort(B.key_array(ctx.s[k], key, pool[k]))
                size = np.searchsorted(pk, qk[m], "right") - np.searchsorted(pk, qk[m], "left")
                eq[m] = e
                dropped[m] = e & (size > spec["cap"])
            ex["+".join(key)] = {"key_equal": int(eq.sum()), "key_equal_but_block_dropped_by_cap": int(dropped.sum())}
        r["exact_channels"] = ex
        # ---- targeted retrieval re-runs (miss S1 + random sample)
        qpos = np.unique(np.concatenate([np.unique(mq), sample_pos]))
        rows = qv[qpos]
        cand_q = cand >> B.Q_SHIFT
        in_sample = np.isin(cand_q, sample_pos)
        cand_sample = cand[in_sample]
        rv = {}
        rank_of = {}
        for ch, k, mp in VARIANTS:
            tag = f"{ch}@{k}/mp{mp}"
            codes_l, ranks_l = [], []
            for src in (2, 3):
                c, rk, _ = B.retrieval_channel(ctx.s[1], rows, ctx.s[src], pool[src], src, ch, k, mp)
                q, s, t = B.decode(c)
                codes_l.append(B.encode(qpos[q], s, t))
                ranks_l.append(rk)
            codes = np.concatenate(codes_l)
            ranks = np.concatenate(ranks_l)
            o = np.argsort(codes)
            codes, ranks = codes[o], ranks[o]
            pos = np.searchsorted(codes, miss)
            pos = np.minimum(pos, len(codes) - 1)
            found = codes[pos] == miss
            mrank = np.where(found, ranks[pos], 10 ** 6)
            rank_of[tag] = mrank
            ks = [kk for kk in KS if kk <= k]
            samp = np.isin(codes >> B.Q_SHIFT, sample_pos)
            rv[tag] = {"missed_recovered_at_k": {str(kk): int((mrank < kk).sum()) for kk in ks},
                       "sanity_missed_with_rank_lt_10_in_production_channel": int((mrank < 10).sum()) if (ch in ("nameaddr", "addr") and mp == 20_000) else None,
                       "new_candidates_per_s1_at_k": {str(kk): float((~np.isin(codes[samp & (ranks < kk)], cand_sample, assume_unique=False)).sum() / SAMPLE) for kk in ks}}
            log(f"  {env} {tag}: recovered " + ", ".join(f"@{kk} {rv[tag]['missed_recovered_at_k'][str(kk)]:,}" for kk in ks))
        r["retrieval"] = rv
        # union of the most promising single additions
        combos = {"nameaddr K=20": rank_of["nameaddr@50/mp20000"] < 20, "nameaddr K=50": rank_of["nameaddr@50/mp20000"] < 50,
                  "addr K=20": rank_of["addr@50/mp20000"] < 20, "name K=10 (new)": rank_of["name@50/mp20000"] < 10,
                  "name K=20 (new)": rank_of["name@50/mp20000"] < 20, "nameaddr budget 200k": rank_of["nameaddr@10/mp200000"] < 10}
        # ---- failure modes (mutually exclusive, priority order)
        modes = [("target name non-Latin (script differs)", nonlatin),
                 ("target address empty", addr_empty & ~nonlatin),
                 ("name shares no token (noisy/domain/different name)", no_name_tok & ~nonlatin & ~addr_empty),
                 ("house number differs", house_diff & ~no_name_tok & ~nonlatin & ~addr_empty),
                 ("address dissimilar (addr_cos < 0.3)", (f("addr_cos") < 0.3) & ~house_diff & ~no_name_tok & ~nonlatin & ~addr_empty)]
        used = np.zeros(len(miss), dtype=bool)
        fm = {}
        for name, m in modes:
            m = m & ~used
            used |= m
            fm[name] = {"missed": int(m.sum()), "share": float(m.mean()),
                        "recovered_by": {cn: int((cm & m).sum()) for cn, cm in combos.items()}}
        rest = ~used
        fm["other (partial name/address differences)"] = {"missed": int(rest.sum()), "share": float(rest.mean()),
                                                          "recovered_by": {cn: int((cm & rest).sum()) for cn, cm in combos.items()}}
        r["failure_modes"] = fm
        r["combos_total_recovered"] = {cn: int(cm.sum()) for cn, cm in combos.items()}
        r["combos_new_candidates_per_s1"] = {
            "nameaddr K=20": rv["nameaddr@50/mp20000"]["new_candidates_per_s1_at_k"]["20"],
            "nameaddr K=50": rv["nameaddr@50/mp20000"]["new_candidates_per_s1_at_k"]["50"],
            "addr K=20": rv["addr@50/mp20000"]["new_candidates_per_s1_at_k"]["20"],
            "name K=10 (new)": rv["name@50/mp20000"]["new_candidates_per_s1_at_k"]["10"],
            "name K=20 (new)": rv["name@50/mp20000"]["new_candidates_per_s1_at_k"]["20"],
            "nameaddr budget 200k": rv["nameaddr@10/mp200000"]["new_candidates_per_s1_at_k"]["10"]}
        r["current_candidates_per_s1"] = float(len(cand) / nq)
        r["seconds"] = round(time.time() - te, 1)
        res["envs"][env] = r
        (OUT / "results.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
        log(f"  env {env} done ({r['seconds']}s): missed {r['missed']:,}; combos {r['combos_total_recovered']}")
        del cand
    ha = {f: sha256(ROOT / f) for f in PROTECTED}
    haf = hash_tree(EXP / "final")
    res["integrity"] = {"protected_unchanged": ha == hb, "experiments_final_unchanged": haf == hbf,
                        "n_protected": len(ha), "n_final_files": len(haf),
                        **{k: ha[k] for k in ("output/matching_results.tsv", "output/candidate_pairs.tsv",
                                              "camelcase_submission.zip")},
                        "experiments/final/model_final.npz": haf.get("experiments/final/model_final.npz")}
    res["seconds"] = round(time.time() - t0, 1)
    (OUT / "results.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    log(f"  integrity {res['integrity']}; total {res['seconds']}s")


if __name__ == "__main__":
    main()
