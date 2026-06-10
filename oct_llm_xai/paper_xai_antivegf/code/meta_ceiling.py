"""Metadata information-ceiling baseline (Part A of the metadata ablation).

How much of the continue/stop DECISION and the ΔCST PROGNOSIS can be predicted from
the pre-treatment METADATA ALONE (age, gender, drug, preVA, preCST) — no image?
This bounds the signal that the VLM metadata-prompt ablation could ever exploit:

  decision   : LogisticRegression / GradientBoosting -> continue_injection
               (balanced accuracy + ROC-AUC, vs the all-continue prior)
  prognosis  : same features -> ΔCST 4-bucket (marked/partial/minimal/worsening)
               (accuracy vs the majority-class baseline)

Same eye-level train/test split as the VLM eval (metadata_v2 'split'; 183/35).
If even a tabular model can't predict ΔCST, the single-timepoint limit is in the
DATA, not the VLM. Env: aptos2021 (sklearn). Out -> sft_data/meta_ceiling.json.
"""
from __future__ import annotations
import json
import numpy as np

ROOT = "/home/ubuntu/bionexus/jgy/OCT_LLM_XAI/oct_llm_xai/paper_xai_antivegf"
META = f"{ROOT}/fluid_masks_v2/metadata_v2.json"
STEROIDS = {"Tricort", "Ozurdex", "Ozrudex"}
PROG = ["marked_response", "partial_response", "minimal_response", "worsening"]


def prog_bucket(d):
    if d <= -100: return 0
    if d <= -25: return 1
    if d <= 25: return 2
    return 3


def featurize(rec):
    return [float(rec["age"]),
            1.0 if rec["gender"] == "Male" else 0.0,
            float(rec["pre_va"]),
            float(rec["pre_cst"]),
            1.0 if rec["drug"] == "Avastin" else 0.0,
            1.0 if rec["drug"] in STEROIDS else 0.0]


FEATS = ["age", "gender_male", "pre_va", "pre_cst", "drug_avastin", "drug_steroid"]


def main():
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score, accuracy_score

    meta = json.load(open(META))
    tr = [r for r in meta if r["split"] == "train"]
    te = [r for r in meta if r["split"] == "test"]
    Xtr = np.array([featurize(r) for r in tr]); Xte = np.array([featurize(r) for r in te])
    yc_tr = np.array([int(r["continue_injection"]) for r in tr])
    yc_te = np.array([int(r["continue_injection"]) for r in te])
    yp_tr = np.array([prog_bucket(r["delta_cst"]) for r in tr])
    yp_te = np.array([prog_bucket(r["delta_cst"]) for r in te])
    print(f"train={len(tr)} test={len(te)} | feats={FEATS}", flush=True)

    rep = {"n_train": len(tr), "n_test": len(te), "features": FEATS}

    # ---- DECISION (binary) ----
    rep["decision"] = {}
    cont_prior = yc_te.mean()
    rep["decision"]["test_continue_rate"] = round(float(cont_prior), 3)
    rep["decision"]["majority_acc"] = round(float(max(cont_prior, 1 - cont_prior)), 3)
    for name, clf in [("logreg", make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))),
                      ("gbm", GradientBoostingClassifier(random_state=0))]:
        clf.fit(Xtr, yc_tr)
        pred = clf.predict(Xte)
        proba = clf.predict_proba(Xte)[:, 1]
        auc = roc_auc_score(yc_te, proba) if len(set(yc_te)) > 1 else float("nan")
        rep["decision"][name] = {"balanced_acc": round(float(balanced_accuracy_score(yc_te, pred)), 3),
                                 "roc_auc": round(float(auc), 3)}
        print(f"[decision/{name}] balAcc={rep['decision'][name]['balanced_acc']} AUC={rep['decision'][name]['roc_auc']}", flush=True)

    # ---- PROGNOSIS (4-class ΔCST) ----
    rep["prognosis"] = {}
    maj = np.bincount(yp_te, minlength=4).max() / len(yp_te)
    rep["prognosis"]["majority_acc"] = round(float(maj), 3)
    for name, clf in [("logreg", make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, multi_class="multinomial"))),
                      ("gbm", GradientBoostingClassifier(random_state=0))]:
        clf.fit(Xtr, yp_tr)
        acc = accuracy_score(yp_te, clf.predict(Xte))
        rep["prognosis"][name] = {"acc": round(float(acc), 3)}
        print(f"[prognosis/{name}] acc={round(float(acc),3)} (maj={rep['prognosis']['majority_acc']})", flush=True)

    # ---- which feature carries the signal? (logreg |coef| on standardized feats) ----
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")).fit(Xtr, yc_tr)
    coefs = lr.named_steps["logisticregression"].coef_[0]
    rep["decision_logreg_coefs"] = {f: round(float(c), 3) for f, c in zip(FEATS, coefs)}
    print("[decision coefs]", rep["decision_logreg_coefs"], flush=True)

    json.dump(rep, open(f"{ROOT}/sft_data/meta_ceiling.json", "w"), indent=1)
    print(f"\nout: {ROOT}/sft_data/meta_ceiling.json", flush=True)


if __name__ == "__main__":
    main()
