#!/usr/bin/env python3
"""READ-ONLY leave-one-country-out (LOCO) diagnostic + unlabelled France inspection of the test predictions.

Nothing in production is modified (SHA-256 of protected files recorded before/after). Outputs go to
experiments/diagnostics/leave_one_country_out/. No test labels exist; the France part is descriptive only.

Models (all: Step 3 protocol — A1 rows of the internal fit-fold set A, all positives + 10 % hashed negatives with
weight 10, GBDT 300 trees with the Step 3 parameters; threshold = best official macro F0.5 on A2, one-per-target ON):
  both  the saved Step 3 model (US + India), threshold 0.69 (Step 3 selection) — the baseline
  US    A1 restricted to US S1;    threshold chosen on the US part of A2
  IN    A1 restricted to India S1; threshold chosen on the India part of A2
  USm   US A1 S1 down-sampled (seeded hash) to the number of India A1 S1 — separates country from training size
Evaluation: validation S1 of one country, on the three environments of the t_rank diagnostic (features reused):
  V normal validation, T test-ratio pool (S1/pool = test 0.1738), S full training pool (stress).
The one-per-target rule is applied within the evaluated country's pairs.

France (unlabelled): current test candidates / target context (experiments/test_scale), final-model probabilities and
submitted predictions (experiments/final), Step 3-model test probabilities; compared with labelled US/India reference
statistics of the Step 3 model on V and T.

Phases: hashes_before, setup, train, select, predict, evaluate, france, hashes_after, report
"""

import argparse
import ctypes
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from ber import blocking as B  # noqa: E402
from ber import match_features as MF  # noqa: E402
from ber import matcher as M  # noqa: E402
from ber.features import FeatureStore  # noqa: E402
from ber.gbdt import GBDT  # noqa: E402
from ber.normalize import h64  # noqa: E402
from ber.validation_split import seeded_hash  # noqa: E402
from matching_experiment import GBDT_PARAMS, NEG_RATE, THRESHOLDS  # noqa: E402
from test_scale_check import jdump, jload, sha256  # noqa: E402
from trank_diagnostic import PROTECTED, log, mem  # noqa: E402

SEED = 20260925
MD = ROOT / "experiments" / "matching"
SD = ROOT / "experiments" / "test_scale" / "stress"
TD = ROOT / "experiments" / "diagnostics" / "trank"
TS = ROOT / "experiments" / "test_scale"
FD = ROOT / "experiments" / "final"
TEST_S1_TSV = ROOT / "dataset" / "test" / "test_source1.tsv"
C = MF.COL
CTRY = {"US": h64("us"), "India": h64("india"), "France": h64("france")}
NEW_MODELS = ["US", "IN", "USm"]
MODELS = ["both"] + NEW_MODELS
TRAIN_CTRY = {"US": "US", "IN": "India", "USm": "US"}
ENVS = {"V": (MD / "cand_val.npy", MD / "X_val.npy"),
        "T": (TD / "cand_T.npy", TD / "X_T.npy"),
        "S": (SD / "cand_stress.npy", TD / "X_S.npy")}
ORACLE = np.round(np.arange(0.50, 0.991, 0.02), 2)
PROB_BINS = [(0.69, 0.80), (0.80, 0.95), (0.95, 1.01)]


def keep_awake():
    """Ask Windows not to sleep while this process runs (per-process request; no system setting is changed)."""
    try:
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
    except Exception:
        pass


class Loco:
    def __init__(self, a):
        from blocking_experiment import Context
        self.a = a
        self.out = Path(a.out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.ctx = Context(a.feat_dir, a.split_dir, a.cache_dir)
        self.qv = np.flatnonzero(self.ctx.fold[1] == 1)
        self.c1 = np.asarray(self.ctx.s[1].col("country"))
        self.cv = self.c1[self.qv]
        self._true = None

    def p(self, n):
        return self.out / n

    @property
    def true(self):
        if self._true is None:
            self._true = self.ctx.truth(self.qv)
        return self._true

    def prob(self, env, m):
        return np.load(TD / f"p_{env}_A.npy") if m == "both" else np.load(self.p(f"p_{env}_{m}.npy"))

    # -----------------------------------------------------------------------------------------------------------
    def hashes_before(self):
        jdump(self.p("hashes_before.json"), {f: sha256(ROOT / f) for f in PROTECTED})

    def hashes_after(self):
        hb = jload(self.p("hashes_before.json"))
        ha = {f: sha256(ROOT / f) for f in hb}
        ok = all(ha[f] == hb[f] for f in hb)
        jdump(self.p("hashes_after.json"), {"hashes": ha, "unchanged": ok, "n_files": len(hb)})
        log(f"  protected files unchanged: {ok} ({len(hb)} files)")

    # -----------------------------------------------------------------------------------------------------------
    def setup(self):
        qA = np.load(MD / "qA_rows.npy")
        a2 = np.load(MD / "a2_mask.npy")
        cA = self.c1[qA]
        masks = {"US": ~a2 & (cA == CTRY["US"]), "IN": ~a2 & (cA == CTRY["India"])}
        idx_us = np.flatnonzero(masks["US"])
        ids1 = np.asarray(self.ctx.s[1].col("id"))
        h = seeded_hash([f"S1-{int(x)}" for x in ids1[qA[idx_us]]], SEED + 900)
        usm = np.zeros(len(qA), dtype=bool)
        usm[idx_us[np.argsort(h, kind="stable")[:int(masks["IN"].sum())]]] = True
        masks["USm"] = usm
        codes = np.load(MD / "cand_A.npy")
        y = np.load(MD / "y_A.npy")
        q = codes >> B.Q_SHIFT
        info = {}
        for m, mk in masks.items():
            np.save(self.p(f"a1_mask_{m}.npy"), mk)
            sel = mk[q]
            info[m] = {"a1_s1": int(mk.sum()), "a1_pairs": int(sel.sum()), "a1_positive_pairs": int(y[sel].sum())}
        for c in ("US", "India"):
            info[f"A2_{c}_s1"] = int((a2 & (cA == CTRY[c])).sum())
        info["validation_s1"] = {c: int((self.cv == CTRY[c]).sum()) for c in ("US", "India")}
        jdump(self.p("setup.json"), info)
        log(f"  setup {info}")

    # -----------------------------------------------------------------------------------------------------------
    def train(self):
        codes = np.load(MD / "cand_A.npy")
        y = np.load(MD / "y_A.npy")
        a2 = np.load(MD / "a2_mask.npy")
        qA = np.load(MD / "qA_rows.npy")
        cA = self.c1[qA]
        q = codes >> B.Q_SHIFT
        with np.errstate(over="ignore"):
            hsh = ((codes.view(np.uint64) * np.uint64(0x9E3779B97F4A7C15)) >> np.uint64(40)) % np.uint64(1000)
        keep_neg = hsh < np.uint64(int(NEG_RATE * 1000))
        del hsh
        XA = np.load(MD / "X_A.npy", mmap_mode="r")
        info = jload(self.p("train.json")) if self.p("train.json").exists() else {}
        for m in NEW_MODELS:
            if self.p(f"pA2_{m}.npy").exists():
                continue
            t0 = time.time()
            mk = np.load(self.p(f"a1_mask_{m}.npy"))
            rows = np.flatnonzero(mk[q] & ((y == 1) | keep_neg))
            w = np.where(y[rows] == 1, 1.0, 1.0 / NEG_RATE)
            gb = GBDT(**GBDT_PARAMS).fit(np.asarray(XA[rows]), y[rows], w, log=log)
            gb.save(self.p(f"model_{m}.npz"))
            a2c = a2 & (cA == CTRY[TRAIN_CTRY[m]])
            rows2 = np.flatnonzero(a2c[q])
            pa2 = np.empty(len(rows2), dtype=np.float32)
            for a in range(0, len(rows2), 2_000_000):
                pa2[a:a + 2_000_000] = gb.predict_proba(np.asarray(XA[rows2[a:a + 2_000_000]]))
            np.save(self.p(f"pA2_{m}.npy"), pa2)
            gi = gb.gain_importance / gb.gain_importance.sum()
            info[m] = {"seconds": round(time.time() - t0, 1), "train_rows": int(len(rows)),
                       "train_pos": int(y[rows].sum()), "a1_s1": int(mk.sum()),
                       "gain_top8": sorted(zip(MF.FEATURE_NAMES, gi.tolist()), key=lambda x: -x[1])[:8],
                       "target_context_gain": float(gi[[C["t_rank"], C["t_gap"], C["t_ncand"]]].sum()),
                       "memory": mem()}
            jdump(self.p("train.json"), info)
            log(f"  model {m} trained ({info[m]['seconds']}s, {len(rows):,} rows) top {info[m]['gain_top8'][:3]}")
        gbA = GBDT.load(MD / "model_gbdt.npz")
        gi = gbA.gain_importance / gbA.gain_importance.sum()
        info["both"] = {"saved_step3_model": True,
                        "gain_top8": sorted(zip(MF.FEATURE_NAMES, gi.tolist()), key=lambda x: -x[1])[:8],
                        "target_context_gain": float(gi[[C["t_rank"], C["t_gap"], C["t_ncand"]]].sum())}
        jdump(self.p("train.json"), info)

    def select(self):
        qA = np.load(MD / "qA_rows.npy")
        a2 = np.load(MD / "a2_mask.npy")
        cA = self.c1[qA]
        codes = np.load(MD / "cand_A.npy")
        q = codes >> B.Q_SHIFT
        trueA = self.ctx.truth(qA)
        res = {"both": {"threshold": 0.69, "note": "Step 3 selection on the full A2 (US + India)"}}
        for m in NEW_MODELS:
            a2c = a2 & (cA == CTRY[TRAIN_CTRY[m]])
            c2 = codes[a2c[q]]
            t2 = trueA[a2c[trueA >> B.Q_SHIFT]]
            idx = np.flatnonzero(a2c)
            p = np.load(self.p(f"pA2_{m}.npy"))
            curve = []
            for th in THRESHOLDS:
                _, parts = M.entity_f05(c2[M.select(c2, p, th, True)], t2, len(qA), return_parts=True)
                curve.append(float(parts["F"][idx].mean()))
            b = int(np.argmax(curve))
            res[m] = {"threshold": float(THRESHOLDS[b]), "a2_f05_own_country": curve[b],
                      "a2_country": TRAIN_CTRY[m]}
            log(f"  {m}: A2({TRAIN_CTRY[m]}) best F0.5 {curve[b]:.5f} at t={THRESHOLDS[b]:.2f}")
        jdump(self.p("select.json"), res)

    # -----------------------------------------------------------------------------------------------------------
    def predict(self):
        models = {m: GBDT.load(self.p(f"model_{m}.npz")) for m in NEW_MODELS}
        for env, (cpath, xpath) in ENVS.items():
            if all(self.p(f"p_{env}_{m}.npy").exists() for m in NEW_MODELS):
                continue
            X = np.load(xpath, mmap_mode="r")
            n = X.shape[0]
            probs = {m: np.empty(n, dtype=np.float32) for m in NEW_MODELS}
            t0 = time.time()
            for a in range(0, n, 2_000_000):
                blk = np.asarray(X[a:a + 2_000_000], dtype=np.float32)
                for m, g in models.items():
                    probs[m][a:a + len(blk)] = g.predict_proba(blk)
            for m in NEW_MODELS:
                np.save(self.p(f"p_{env}_{m}.npy"), probs[m])
            log(f"  predicted {env} ({n:,} pairs, {time.time() - t0:.0f}s) {mem()}")

    # -----------------------------------------------------------------------------------------------------------
    def _target_origin(self, codes):
        _, s, t = B.decode(codes)
        org = np.zeros(len(codes), dtype=np.int8)
        for k in (2, 3):
            is_t = np.zeros(self.ctx.s[k].n, dtype=bool)
            is_t[self.ctx.pair_trow[self.ctx.pair_src == k]] = True
            m = s == k
            fit = self.ctx.fold[k][t[m]] == 0
            o = np.zeros(int(m.sum()), dtype=np.int8)
            o[fit & is_t[t[m]]] = 1
            o[fit & ~is_t[t[m]]] = 2
            org[m] = o
        return org

    def evaluate(self):
        sel = jload(self.p("select.json"))
        true, nq = self.true, len(self.qv)
        out = jload(self.p("evaluate.json")) if self.p("evaluate.json").exists() else {}
        for env, (cpath, _) in ENVS.items():
            if env in out:
                continue
            codes = np.load(cpath)
            y = np.isin(codes, true, assume_unique=True)
            q = codes >> B.Q_SHIFT
            cols = np.load(TD / f"cols_{env}.npz")
            org = self._target_origin(codes)
            probs = {m: self.prob(env, m) for m in MODELS}
            out[env] = {}
            for ec in ("US", "India"):
                qmask = self.cv == CTRY[ec]
                pm = qmask[q]
                cs, ys = codes[pm], y[pm]
                tr, tn, og = cols["t_rank"][pm], cols["t_ncand"][pm], org[pm]
                true_c = true[qmask[true >> B.Q_SHIFT]]
                ntrue_c = np.bincount(true_c >> B.Q_SHIFT, minlength=nq)[qmask]
                block = {"n_s1": int(qmask.sum()), "n_singletons": int((ntrue_c == 0).sum()),
                         "true_pairs": int(len(true_c)), "candidate_pairs": int(len(cs)),
                         "candidate_recall": float(ys.sum() / len(true_c)),
                         "t_rank0_share_of_candidates": float((tr == 0).mean()),
                         "uncontested_share_of_candidates": float((tn == 1).mean()), "models": {}}
                for m in MODELS:
                    p = probs[m][pm]
                    th = sel[m]["threshold"]
                    keep = M.select(cs, p, th, True)
                    _, parts = M.entity_f05(cs[keep], true_c, nq, return_parts=True)
                    F = parts["F"][qmask]
                    single = parts["ntrue"][qmask] == 0
                    tp = int((keep & ys).sum())
                    fpm = keep & ~ys
                    npred = int(keep.sum())
                    r = {"threshold": th, "macro_f05": float(F.mean()),
                         "precision": tp / npred if npred else 0.0, "recall": tp / len(true_c),
                         "singleton_f05": float(F[single].mean()), "nonsingleton_f05": float(F[~single].mean()),
                         "singleton_false_merge_rate": float((parts["npred"][qmask][single] > 0).mean()),
                         "zero_prediction_rate": float((parts["npred"][qmask] == 0).mean()),
                         "pred": npred, "tp": tp, "fp": int(fpm.sum()), "fn": int(len(true_c) - tp),
                         "fp_t_rank0": int((fpm & (tr == 0)).sum()),
                         "fp_t_rank0_uncontested": int((fpm & (tr == 0) & (tn == 1)).sum()),
                         "fp_orphan_target": int((fpm & (og == 1)).sum()),
                         "fp_fit_distractor": int((fpm & (og == 2)).sum()),
                         "precision_t_rank0": float((keep & ys & (tr == 0)).sum() / max(1, (keep & (tr == 0)).sum()))}
                    pre = np.flatnonzero(p >= ORACLE[0])
                    c2, p2 = cs[pre], p[pre]
                    best = (-1.0, None)
                    for t in ORACLE:
                        _, pp = M.entity_f05(c2[M.select(c2, p2, t, True)], true_c, nq, return_parts=True)
                        f = float(pp["F"][qmask].mean())
                        if f > best[0]:
                            best = (f, float(t))
                    r["oracle_threshold"], r["oracle_macro_f05"] = best[1], best[0]
                    block["models"][m] = r
                    log(f"  {env}/{ec}/{m}: F0.5 {r['macro_f05']:.5f} P {r['precision']:.4f} R {r['recall']:.4f} "
                        f"single {r['singleton_f05']:.4f} FP {r['fp']:,} FN {r['fn']:,} | oracle t={best[1]} "
                        f"{best[0]:.5f}")
                out[env][ec] = block
            jdump(self.p("evaluate.json"), out)

    # -----------------------------------------------------------------------------------------------------------
    def france(self):
        """Unlabelled per-country statistics of the current test predictions + labelled US/India reference."""
        S1 = FeatureStore(TS / "features_test", "test_s1")
        pools = {2: FeatureStore(TS / "features_test", "test_s2"), 3: FeatureStore(TS / "features_test", "test_s3")}
        n1 = S1.n
        c1 = np.asarray(S1.col("country"))
        cand = np.load(TS / "cand_test.npy")
        q_all = cand >> B.Q_SHIFT
        starts = np.searchsorted(q_all, np.arange(n1))
        tr = np.load(TS / "t_rank_test.npy")
        tn = np.load(TS / "t_ncand_test.npy")
        p_fin = np.load(FD / "test_prob.npy")
        p_s3 = np.load(TS / "prob_test.npy")
        pred_fin = np.load(FD / "test_pred_codes.npy")
        idx_fin = np.searchsorted(cand, pred_fin)
        assert (cand[idx_fin] == pred_fin).all()
        keep_s3 = M.select(cand, p_s3, 0.69, True)
        idx_s3 = np.flatnonzero(keep_s3)
        del keep_s3
        idf_n, idf_a = (MF.IdfTable.load(TS / f"idf_test_{c}.npz") for c in ("name", "addr"))

        def pair_cols(idx):
            """name_cos, addr_cos, house conflict, target addr missing, eq_name_n2, legal-only for pairs cand[idx]."""
            out = {k: np.empty(len(idx), dtype=np.float32) for k in
                   ("name_cos", "addr_cos", "house_conflict", "t_addr_missing", "eq_name_n2", "legal_only")}
            for a in range(0, len(idx), 1_000_000):
                cc = cand[idx[a:a + 1_000_000]]
                X = MF.pair_features(S1, np.arange(n1), pools, cc, idf_n, idf_a)
                sl = slice(a, a + len(cc))
                out["name_cos"][sl] = X[:, C["name_cos"]]
                out["addr_cos"][sl] = X[:, C["addr_cos"]]
                out["house_conflict"][sl] = (X[:, C["house_both"]] == 1) & (X[:, C["eq_house"]] == 0)
                out["t_addr_missing"][sl] = X[:, C["t_addr_missing"]]
                out["eq_name_n2"][sl] = X[:, C["eq_name_n2"]]
                out["legal_only"][sl] = (X[:, C["eq_name_n2"]] == 1) & (X[:, C["eq_name_n1"]] == 0)
            return out

        def categories(tr_, tn_, pc, prob_):
            cats = {"t_rank0 & uncontested (t_ncand=1)": (tr_ == 0) & (tn_ == 1),
                    "t_rank0 & contested": (tr_ == 0) & (tn_ > 1),
                    "same name / different address": (pc["name_cos"] >= 0.8) & (pc["addr_cos"] < 0.3),
                    "house-number conflict": pc["house_conflict"] == 1,
                    "target address empty": pc["t_addr_missing"] == 1}
            for lo, hi in PROB_BINS:
                cats[f"prob [{lo:.2f},{min(hi, 1.0):.2f})"] = (prob_ >= lo) & (prob_ < hi)
            return cats

        res = {"per_country": {}, "labelled_reference_step3_model": {}}
        pcs = {"final": pair_cols(idx_fin), "step3": pair_cols(idx_s3)}
        name_n2 = np.asarray(S1.col("name_n2"))
        ids_q_fin, ids_q_s3 = q_all[idx_fin], q_all[idx_s3]
        maxp = np.maximum.reduceat(p_fin, starts)
        top_names = {}
        for c, hc in CTRY.items():
            sm = c1 == hc
            ns = int(sm.sum())
            npf = np.bincount(ids_q_fin, minlength=n1)
            nps = np.bincount(ids_q_s3, minlength=n1)
            ncand = np.diff(np.append(starts, len(cand)))
            nm = name_n2[sm]
            u, inv, cnt = np.unique(nm, return_inverse=True, return_counts=True)
            per = cnt[inv]
            gen = per >= 10
            r = {"n_s1": ns, "candidates_per_s1": float(ncand[sm].mean()),
                 "final": {"pred_per_s1": float(npf[sm].mean()), "zero_prediction_rate": float((npf[sm] == 0).mean()),
                           "s1_with_3plus_predictions": float((npf[sm] >= 3).mean())},
                 "step3_model": {"pred_per_s1": float(nps[sm].mean()),
                                 "zero_prediction_rate": float((nps[sm] == 0).mean())},
                 "max_candidate_prob_final": {"<0.10": float((maxp[sm] < 0.10).mean()),
                                              "[0.10,0.69)": float(((maxp[sm] >= 0.10) & (maxp[sm] < 0.69)).mean()),
                                              ">=0.69": float((maxp[sm] >= 0.69).mean())},
                 "generic_names": {"share_s1_name_shared_by_2+": float((per >= 2).mean()),
                                   "share_s1_name_shared_by_10+": float(gen.mean()),
                                   "share_s1_name_shared_by_100+": float((per >= 100).mean()),
                                   "distinct_names_per_s1": float(len(u) / ns),
                                   "zero_pred_rate_generic(10+)": float((npf[sm][gen] == 0).mean()) if gen.any() else None,
                                   "zero_pred_rate_unique(1)": float((npf[sm][per == 1] == 0).mean()),
                                   "pred_per_s1_generic(10+)": float(npf[sm][gen].mean()) if gen.any() else None,
                                   "pred_per_s1_unique(1)": float(npf[sm][per == 1].mean())}}
            order = np.argsort(-cnt, kind="stable")[:10]
            rows_sm = np.flatnonzero(sm)
            top_names[c] = [(int(rows_sm[np.flatnonzero(inv == j)[0]]), int(cnt[j])) for j in order]
            for tag, idx, pc, pq, pr in (("final", idx_fin, pcs["final"], ids_q_fin, p_fin),
                                         ("step3_model", idx_s3, pcs["step3"], ids_q_s3, p_s3)):
                m = sm[pq]
                cats = categories(tr[idx][m], tn[idx][m], {k: v[m] for k, v in pc.items()}, pr[idx][m])
                r[tag]["prediction_category_shares"] = {k: float(v.mean()) for k, v in cats.items()}
                if tag == "final":
                    r[tag]["legal_form_only_name_difference_share"] = float(pc["legal_only"][m].mean())
                    r[tag]["eq_name_n2_share"] = float(pc["eq_name_n2"][m].mean())
            res["per_country"][c] = r
            log(f"  {c}: {r['final']['pred_per_s1']:.3f} pred/S1, zero {r['final']['zero_prediction_rate']:.4f}, "
                f"generic10+ {r['generic_names']['share_s1_name_shared_by_10+']:.4f}")
        # raw example names (from the test S1 file; names only, first occurrence of the 10 most frequent per country)
        need = {row for v in top_names.values() for row, _ in v}
        got = {}
        with open(TEST_S1_TSV, encoding="utf-8") as f:
            next(f)
            for i, line in enumerate(f):
                if i in need:
                    got[i] = line.rstrip("\n").split("\t")[1]
                if len(got) == len(need):
                    break
        res["most_frequent_s1_names"] = {c: [(got[row], n) for row, n in v] for c, v in top_names.items()}
        # labelled reference on validation environments (Step 3 model, same categories, with precision)
        for env in ("V", "T"):
            codes = np.load(ENVS[env][0])
            y = np.isin(codes, self.true, assume_unique=True)
            pA = self.prob(env, "both")
            cols = np.load(TD / f"cols_{env}.npz")
            q = codes >> B.Q_SHIFT
            res["labelled_reference_step3_model"][env] = {}
            for ec in ("US", "India"):
                pm = (self.cv == CTRY[ec])[q]
                cs = codes[pm]
                keep = M.select(cs, pA[pm], 0.69, True)
                ii = np.flatnonzero(pm)[keep]
                pc = {"name_cos": cols["name_cos"][ii], "addr_cos": cols["addr_cos"][ii],
                      "house_conflict": ((cols["house_both"][ii] == 1) & (cols["eq_house"][ii] == 0)).astype(np.float32),
                      "t_addr_missing": cols["t_addr_missing"][ii]}
                cats = categories(cols["t_rank"][ii], cols["t_ncand"][ii], pc, pA[ii])
                yy = y[ii]
                npred = np.bincount(q[ii], minlength=len(self.qv))[self.cv == CTRY[ec]]
                res["labelled_reference_step3_model"][env][ec] = {
                    "pred_per_s1": float(npred.mean()), "zero_prediction_rate": float((npred == 0).mean()),
                    "category_share": {k: float(v.mean()) for k, v in cats.items()},
                    "category_precision": {k: (float(yy[v].mean()) if v.any() else None) for k, v in cats.items()}}
        jdump(self.p("france.json"), res)

    def report(self):
        write_report(self)


def pc(x):
    return "—" if x is None else f"{100 * x:.2f}%"


def write_report(d):
    E, S, Tr, St = (jload(d.p(n)) for n in ("evaluate.json", "select.json", "train.json", "setup.json"))
    Fr = jload(d.p("france.json"))
    H = jload(d.p("hashes_after.json")) if d.p("hashes_after.json").exists() else {}
    L = []
    w = L.append
    w("# Leave-one-country-out diagnostic + France test inspection (read-only)\n")
    w("Generated by `experiments/loco_diagnostic.py`. Training labels only; no test labels exist. No production file "
      "modified; no model selected; no submission produced.\n")
    w("## Models\n")
    w("| model | training S1 (A1) | training rows | A2 threshold (own country) | A2 F0.5 | target-context gain | top gain |")
    w("|---|---:|---:|---:|---:|---:|---|")
    for m in MODELS:
        t = Tr[m]
        w(f"| {m} | {t.get('a1_s1', 353091):,} | {t.get('train_rows', 3142902):,} | {S[m]['threshold']:.2f} | "
          f"{S[m].get('a2_f05_own_country', 0.96289):.5f} | {pc(t['target_context_gain'])} | "
          + ", ".join(f"`{n}` {100 * g:.1f}%" for n, g in t["gain_top8"][:3]) + " |")
    w(f"\nSetup: {St}\n")
    w("## Results by evaluation country\n")
    w("| env | eval country | model | thr | macro F0.5 | precision | recall | singleton F0.5 | zero-pred | FP | FN | "
      "FP t_rank=0 | FP t_rank=0 uncontested | FP orphan | oracle t | oracle F0.5 |")
    w("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for env in ("V", "T", "S"):
        for ec in ("US", "India"):
            for m in MODELS:
                r = E[env][ec]["models"][m]
                w(f"| {env} | {ec} | {m} | {r['threshold']:.2f} | **{r['macro_f05']:.5f}** | {pc(r['precision'])} | "
                  f"{pc(r['recall'])} | {pc(r['singleton_f05'])} | {pc(r['zero_prediction_rate'])} | {r['fp']:,} | "
                  f"{r['fn']:,} | {r['fp_t_rank0']:,} | {r['fp_t_rank0_uncontested']:,} | {r['fp_orphan_target']:,} | "
                  f"{r['oracle_threshold']} | {r['oracle_macro_f05']:.5f} |")
    w("\nCandidate recall by country: " + "; ".join(
        f"{env}/{ec} {pc(E[env][ec]['candidate_recall'])}" for env in ("V", "T", "S") for ec in ("US", "India")) + "\n")
    w("## Transfer penalty (in-country minus cross-country, macro F0.5)\n")
    w("| env | eval US: USm − IN (size-matched) | eval US: US − IN | eval India: IN − USm (size-matched) | eval India: IN − US |")
    w("|---|---:|---:|---:|---:|")
    for env in ("V", "T", "S"):
        g = lambda ec, m: E[env][ec]["models"][m]["macro_f05"]  # noqa: E731
        w(f"| {env} | {g('US', 'USm') - g('US', 'IN'):+.5f} | {g('US', 'US') - g('US', 'IN'):+.5f} | "
          f"{g('India', 'IN') - g('India', 'USm'):+.5f} | {g('India', 'IN') - g('India', 'US'):+.5f} |")
    w("\n## France (unlabelled test) vs US / India\n")
    w("| statistic | US | India | France |")
    w("|---|---:|---:|---:|")
    P = Fr["per_country"]
    rows = [("S1", lambda r: f"{r['n_s1']:,}"), ("candidates / S1", lambda r: f"{r['candidates_per_s1']:.1f}"),
            ("final: predictions / S1", lambda r: f"{r['final']['pred_per_s1']:.3f}"),
            ("final: zero-prediction S1", lambda r: pc(r['final']['zero_prediction_rate'])),
            ("Step 3 model: predictions / S1", lambda r: f"{r['step3_model']['pred_per_s1']:.3f}"),
            ("Step 3 model: zero-prediction S1", lambda r: pc(r['step3_model']['zero_prediction_rate'])),
            ("S1 whose best candidate prob < 0.10", lambda r: pc(r['max_candidate_prob_final']['<0.10'])),
            ("S1 whose best candidate prob in [0.10,0.69)", lambda r: pc(r['max_candidate_prob_final']['[0.10,0.69)'])),
            ("S1 name shared by ≥2 S1 (same country)", lambda r: pc(r['generic_names']['share_s1_name_shared_by_2+'])),
            ("S1 name shared by ≥10 S1", lambda r: pc(r['generic_names']['share_s1_name_shared_by_10+'])),
            ("S1 name shared by ≥100 S1", lambda r: pc(r['generic_names']['share_s1_name_shared_by_100+'])),
            ("zero-pred rate: generic (≥10) / unique names",
             lambda r: f"{pc(r['generic_names']['zero_pred_rate_generic(10+)'])} / {pc(r['generic_names']['zero_pred_rate_unique(1)'])}"),
            ("pred/S1: generic (≥10) / unique names",
             lambda r: f"{r['generic_names']['pred_per_s1_generic(10+)'] or 0:.3f} / {r['generic_names']['pred_per_s1_unique(1)']:.3f}"),
            ("final preds: legal-form-only name difference", lambda r: pc(r['final']['legal_form_only_name_difference_share']))]
    for lab, f in rows:
        w(f"| {lab} | " + " | ".join(f(P[c]) for c in ("US", "India", "France")) + " |")
    for k in P["US"]["final"]["prediction_category_shares"]:
        w(f"| final preds: {k} | " + " | ".join(pc(P[c]['final']['prediction_category_shares'][k]) for c in ("US", "India", "France")) + " |")
    w("\nLabelled reference (Step 3 model, validation S1) — share of predictions per category and precision; compare "
      "with the Step 3 model's test shares below.\n")
    w("| category | V US share / prec. | V India share / prec. | T US share / prec. | T India share / prec. | test US share | test India share | test France share |")
    w("|---|---:|---:|---:|---:|---:|---:|---:|")
    R = Fr["labelled_reference_step3_model"]
    for k in R["V"]["US"]["category_share"]:
        cells = [f"{pc(R[e][c]['category_share'][k])} / {pc(R[e][c]['category_precision'][k])}" for e in ("V", "T") for c in ("US", "India")]
        cells += [pc(P[c]["step3_model"]["prediction_category_shares"][k]) for c in ("US", "India", "France")]
        w(f"| {k} | " + " | ".join(cells) + " |")
    w("\nMost frequent test S1 names per country (first occurrence text, count): ")
    for c, lst in Fr["most_frequent_s1_names"].items():
        w(f"- {c}: " + "; ".join(f"{n} ×{k}" for n, k in lst))
    w(f"\n## Integrity\n\nProtected files unchanged: **{H.get('unchanged')}** ({H.get('n_files')} files).\n")
    (d.out / "LOCO_REPORT.md").write_text("\n".join(L), encoding="utf-8")
    log(f"  report: {d.out / 'LOCO_REPORT.md'}")


PHASES = ["hashes_before", "setup", "train", "select", "predict", "evaluate", "france", "hashes_after", "report"]
PHASE_OUT = {"hashes_before": "hashes_before.json", "setup": "setup.json", "train": "train.done",
             "select": "select.json", "predict": "predict.done", "evaluate": "evaluate.done", "france": "france.json",
             "hashes_after": None, "report": None}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split-dir", default=str(ROOT / "experiments" / "validation_split"))
    ap.add_argument("--cache-dir", default=str(ROOT / "experiments" / "_eda_cache"))
    ap.add_argument("--feat-dir", default=str(ROOT / "experiments" / "blocking" / "features"))
    ap.add_argument("--out-dir", default=str(ROOT / "experiments" / "diagnostics" / "leave_one_country_out"))
    ap.add_argument("--phase", default="all", choices=["all"] + PHASES)
    ap.add_argument("--smoke", action="store_true", help="use the experiments/_smoke copies of every input")
    a = ap.parse_args()
    if a.smoke:
        global MD, SD, TD, TS, FD, TEST_S1_TSV, ENVS
        sm = ROOT / "experiments" / "_smoke"
        MD, SD, TD, TS, FD = (sm / "matching", sm / "test_scale" / "stress", sm / "diagnostics_trank",
                              sm / "test_scale", sm / "final")
        TEST_S1_TSV = sm / "dataset" / "test" / "test_source1.tsv"
        ENVS = {"V": (MD / "cand_val.npy", MD / "X_val.npy"), "T": (TD / "cand_T.npy", TD / "X_T.npy"),
                "S": (SD / "cand_stress.npy", TD / "X_S.npy")}
        a.split_dir, a.cache_dir = str(sm / "validation_split"), str(sm / "_eda_cache")
        a.feat_dir, a.out_dir = str(sm / "blocking" / "features"), str(sm / "diagnostics_loco")
    keep_awake()
    d = Loco(a)
    tp = d.p("timings.json")
    timings = jload(tp) if tp.exists() else {}
    for ph in PHASES:
        if a.phase not in ("all", ph):
            continue
        o = PHASE_OUT[ph]
        if o and d.p(o).exists() and a.phase == "all":
            log(f"[{ph}] reuse")
            continue
        log(f"[{ph}]")
        t0 = time.time()
        getattr(d, ph)()
        if o and o.endswith(".done"):
            d.p(o).write_text("ok", encoding="utf-8")
        timings[ph] = round(time.time() - t0, 1)
        jdump(tp, timings)
        log(f"[{ph}] done ({timings[ph]}s) {mem()}")


if __name__ == "__main__":
    main()
