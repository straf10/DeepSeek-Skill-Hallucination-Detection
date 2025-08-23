import json
import numpy as np
import pandas as pd
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support,
)


# ---------- Paths / Config ----------
INPUT = r"C:\Python\THESIS\skillab_job_fetcher\output\data.xlsx"
OUTPUT_LOG_REG = r"C:\Python\THESIS\skillab_job_fetcher\output\log_reg_results"

def load_data(xlsx_path: str) -> pd.DataFrame:
    df = pd.read_excel(xlsx_path, dtype={"original_skill": str, "manual_label": str, "source": str})

    need = {"original_skill", "manual_label", "source"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)} (expected: {sorted(need)})")

    df["manual_label"] = df["manual_label"].astype(str).str.strip().str.lower().str.replace(r"[^a-z0-9]+", "", regex=True)
    df = df[df["original_skill"].astype(str).str.strip().astype(bool)].reset_index(drop=True)
    return df[["original_skill", "manual_label", "source"]]

def to_binary_labels(y: pd.Series) -> pd.Series:
    # H0 -> H0 | H1x -> H1
    yb = y.apply(lambda z: "h0" if z == "h0" else ("h1" if str(z).startswith("h1") else "other"))
    mask = yb.isin(["h0", "h1"])
    return yb[mask], mask

def build_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 3),
        min_df=1,
        sublinear_tf=True,
        norm="l2",
        lowercase=True,
        stop_words=None,
    )

def train_eval_lr(X_text: pd.Series, y: pd.Series, labels_order: list[str], tag: str):
    """
    Εκπαίδευση + αξιολόγηση Logistic Regression για ένα σενάριο (binary ή multiclass)
    """
    Xtr, Xte, ytr, yte = train_test_split(
        X_text, y, test_size=0.2, stratify=y, random_state=42
    )

    vec = build_vectorizer()
    lr = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        solver="lbfgs",
    )
    pipe = make_pipeline(vec, lr)

    # 5-fold CV (macro-F1) στο train
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipe, Xtr, ytr, scoring="f1_macro", cv=kf, n_jobs=-1)

    # Train + Test
    pipe.fit(Xtr, ytr)
    ypred = pipe.predict(Xte)

    # Metrics
    report_dict = classification_report(yte, ypred, labels=labels_order, output_dict=True, zero_division=0)
    acc = accuracy_score(yte, ypred)
    prec, rec, f1, support = precision_recall_fscore_support(yte, ypred, labels=labels_order, zero_division=0)
    cm = confusion_matrix(yte, ypred, labels=labels_order)

    # Save reports
    out_base = Path(OUTPUT_LOG_REG) / f"lr_{tag}"
    out_base.parent.mkdir(parents=True, exist_ok=True)

    with (out_base.with_suffix(".json")).open("w", encoding="utf-8") as f:
        json.dump({
            "scenario": tag,
            "labels": labels_order,
            "classification_report": report_dict,
            "accuracy": acc,
            "cv_macro_f1_mean": float(np.mean(cv_scores)),
            "cv_macro_f1_std": float(np.std(cv_scores)),
            "confusion_matrix": cm.tolist(),
        }, f, ensure_ascii=False, indent=2)

    # Confusion matrix CSV
    pd.DataFrame(cm, index=labels_order, columns=labels_order).to_csv(out_base.with_suffix(".confusion.csv"), encoding="utf-8", index=True)

    # summary CSV (macro/weighted/micro)
    summary_row = pd.DataFrame([{
        "scenario": tag,
        "model": "logistic_regression",
        "accuracy": acc,
        "cv_macro_f1_mean": float(np.mean(cv_scores)),
        "cv_macro_f1_std": float(np.std(cv_scores)),
        "f1_macro": report_dict["macro avg"]["f1-score"],
        "precision_macro": report_dict["macro avg"]["precision"],
        "recall_macro": report_dict["macro avg"]["recall"],
        "f1_weighted": report_dict["weighted avg"]["f1-score"],
        "precision_weighted": report_dict["weighted avg"]["precision"],
        "recall_weighted": report_dict["weighted avg"]["recall"],
        # Micro-F1 ~ accuracy για πλήρη πρόβλεψη
        "f1_micro": acc,
    }])
    summary_csv = Path(OUTPUT_LOG_REG) / "lr_summary_metrics.csv"
    if summary_csv.exists():
        prev = pd.read_csv(summary_csv)
        pd.concat([prev, summary_row], ignore_index=True).to_csv(summary_csv, index=False)
    else:
        summary_row.to_csv(summary_csv, index=False)

    # Save fitted pipeline (vectorizer + logistic regression) για reuse
    model_path = out_base.with_suffix(".joblib")
    joblib.dump(pipe, model_path)

    return {
        "pipe": pipe,
        "labels_order": labels_order,
        "cv_scores": cv_scores,
        "report": report_dict,
        "confusion": cm,
        "y_true": yte,
        "y_pred": ypred,
        "model_path": str(model_path),
        "report_path": str(out_base.with_suffix(".json")),
    }

def top_features_binary(lr_pipe, n=20):
    """
    Top χαρακτηριστικά για binary (H0 vs H1).
    Επιστρέφει δύο λίστες: προς H1 (θετικά coef) και προς H0 (αρνητικά coef).
    """
    # Unpack vectorizer & LR
    vec: TfidfVectorizer = lr_pipe.named_steps["tfidfvectorizer"]
    lr: LogisticRegression = lr_pipe.named_steps["logisticregression"]
    feature_names = np.array(vec.get_feature_names_out())

    # Για binary, lr.coef_.shape = (1, n_features); θετικά -> class 1
    coefs = lr.coef_[0]
    top_h1_idx = np.argsort(coefs)[::-1][:n]
    top_h0_idx = np.argsort(coefs)[:n]

    return list(zip(feature_names[top_h1_idx], coefs[top_h1_idx])), list(zip(feature_names[top_h0_idx], coefs[top_h0_idx]))

def top_features_multiclass(lr_pipe, labels: list[str], n=10):
    """
    Top χαρακτηριστικά ανά κλάση για multiclass (one-vs-rest / multinomial).
    Επιστρέφει dict: class -> [(token, weight), ...]
    """
    vec: TfidfVectorizer = lr_pipe.named_steps["tfidfvectorizer"]
    lr: LogisticRegression = lr_pipe.named_steps["logisticregression"]
    feature_names = np.array(vec.get_feature_names_out())

    tops = {}
    for i, cls in enumerate(lr.classes_):
        if cls not in labels:  # safety
            continue
        coefs = lr.coef_[i]
        idx = np.argsort(coefs)[::-1][:n]
        tops[cls] = list(zip(feature_names[idx], coefs[idx]))
    return tops

if __name__ == "__main__":
    df = load_data(INPUT)
    X_text = df["original_skill"]
    y_full = df["manual_label"]

    # ---------- Binary (H0 vs H1*)
    y_bin, mask_bin = to_binary_labels(y_full)
    X_bin = X_text[mask_bin].reset_index(drop=True)
    y_bin = y_bin.reset_index(drop=True)
    labels_bin = sorted(y_bin.unique())  # ["h0","h1"]

    res_bin = train_eval_lr(X_bin, y_bin, labels_bin, tag="binary_h0_vs_h1")

    # Αποθήκευση top features
    try:
        top_h1, top_h0 = top_features_binary(res_bin["pipe"], n=25)
        with open(Path(OUTPUT_LOG_REG) / "lr_binary_top_features.json", "w", encoding="utf-8") as f:
            json.dump({
                "towards_h1": top_h1,
                "towards_h0": top_h0
            }, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[WARN] Could not export binary top-features: {e}")

    # ---------- Multiclass (H0 vs H1a…e)
    y_multi = y_full.copy()
    # Κρατάμε όσες κλάσεις έχουν τουλάχιστον 4 δείγματα (για σταθερό CV/split)
    cls_counts = y_multi.value_counts()
    keep_classes = cls_counts[cls_counts >= 4].index.tolist()
    mask_multi = y_multi.isin(keep_classes)
    X_multi = X_text[mask_multi].reset_index(drop=True)
    y_multi = y_multi[mask_multi].reset_index(drop=True)
    labels_multi = sorted(y_multi.unique())

    if len(labels_multi) >= 2:
        res_multi = train_eval_lr(X_multi, y_multi, labels_multi, tag="multiclass_h0_h1subtypes")

        # Top features per class
        try:
            tops = top_features_multiclass(res_multi["pipe"], labels_multi, n=15)
            with open(Path(OUTPUT_LOG_REG) / "lr_multiclass_top_features.json", "w", encoding="utf-8") as f:
                json.dump(tops, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[WARN] Could not export multiclass top-features: {e}")
    else:
        print("[INFO] Multiclass training skipped (not enough classes with sufficient samples).")

    print("\n=== Logistic Regression baselines completed ===")
    print(f"Output folder: {OUTPUT_LOG_REG}")