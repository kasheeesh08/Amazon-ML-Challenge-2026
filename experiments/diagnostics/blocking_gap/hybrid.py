#!/usr/bin/env python3
"""Hybrid estimate (READ-ONLY): current-model predictions in the test-ratio environment T (target context like test)
with the candidate-recall losses of the full-pool environment S (absolute pool size like test).

Every true pair that the unchanged Step 2 blocking did NOT retrieve against the full training pool (S) is removed from
T's predictions (a matcher can never predict a pair that was never a candidate). Nothing else changes; this is an
analysis of saved arrays only (trank p_T_A.npy, cand_T.npy, stress cand_stress.npy).
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from ber import blocking as B  # noqa: E402
from ber import matcher as M  # noqa: E402
from ber.normalize import h64  # noqa: E402
from blocking_experiment import Context  # noqa: E402

OUT = Path(__file__).resolve().parent
TD = ROOT / "experiments" / "diagnostics" / "trank"
SD = ROOT / "experiments" / "test_scale" / "stress"

ctx = Context(ROOT / "experiments" / "blocking" / "features", ROOT / "experiments" / "validation_split",
              ROOT / "experiments" / "_eda_cache")
qv = np.flatnonzero(ctx.fold[1] == 1)
nq = len(qv)
cv = np.asarray(ctx.s[1].col("country"))[qv]
true = ctx.truth(qv)
cT = np.load(TD / "cand_T.npy")
pT = np.load(TD / "p_T_A.npy")
predT = cT[M.select(cT, pT, 0.69, True)]
cS = np.load(SD / "cand_stress.npy")
missed_S = true[~np.isin(true, cS, assume_unique=True)]
missed_T = true[~np.isin(true, cT, assume_unique=True)]
drop = np.isin(predT, missed_S, assume_unique=True)
hyb = predT[~drop]
res = {"T_predictions": int(len(predT)), "true_pairs_missed_by_S_blocking": int(len(missed_S)),
       "true_pairs_missed_by_T_blocking": int(len(missed_T)),
       "T_true_predictions_removed_because_S_blocking_misses_them": int(drop.sum()), "overall": {}, "by_country": {}}
for tag, pr in (("T (current model, as measured)", predT), ("hybrid: T matcher + S blocking losses", hyb)):
    f, parts = M.entity_f05(pr, true, nq, return_parts=True)
    tp = int(np.isin(pr, true, assume_unique=True).sum())
    single = parts["ntrue"] == 0
    res["overall"][tag] = {"macro_f05": f, "pair_precision": tp / len(pr), "pair_recall": tp / len(true),
                           "singleton_f05": float(parts["F"][single].mean())}
    for c in ("US", "India"):
        m = cv == h64(c.lower())
        res["by_country"].setdefault(c, {})[tag] = float(parts["F"][m].mean())
d = res["overall"]
res["blocking_cost_at_test_pool_size"] = d["T (current model, as measured)"]["macro_f05"] - d["hybrid: T matcher + S blocking losses"]["macro_f05"]
(OUT / "hybrid.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
print(json.dumps(res, indent=1))
