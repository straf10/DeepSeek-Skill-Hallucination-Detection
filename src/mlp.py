import json, numpy as np, pandas as pd, joblib
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support

# ---------- Paths / Config ----------
INPUT = r"C:\Python\THESIS\skillab_job_fetcher\output\data.xlsx"
OUTPUT_MLP = r"C:\Python\THESIS\skillab_job_fetcher\output\mlp_results"

def load_data(xlsx_path: str) -> pd.DataFrame:
    df = pd.read_excel(xlsx_path, dtype={"original_skill": str, "manual_label": str, "source": str})
    need = {"original_skill", "manual_label", "source"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)} (expected: {sorted(need)})")
    df["manual_label"] = (
        df["manual_label"].astype(str).str.strip().str.lower()
        .str.replace(r"[^a-z0-9]+", "", regex=True)
    )
    df = df[df["original_skill"].astype(str).str.strip().astype(bool)].reset_index(drop=True)
    return df[["original_skill", "manual_label", "source"]]

def to_binary_labels(y: pd.Series):
    yb = y.apply(lambda z: "h0" if z == "h0" else ("h1" if str(z).startswith("h1") else "other"))
    mask = yb.isin(["h0", "h1"])
    return yb[mask], mask

def build_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(analyzer="word", ngram_range=(1,3), min_df=1, sublinear_tf=True, norm="l2", lowercase=True)

def train_eval_mlp(X_text: pd.Series, y: pd.Series, labels_order: list[str], tag: str):
    Xtr, Xte, ytr, yte = train_test_split(X_text, y, test_size=0.2, stratify=y, random_state=42)

    vec = build_vectorizer()
    mlp = MLPClassifier(
        hidden_layer_sizes=(256, ),
        activation="relu",
        solver="adam",
        alpha=1e-4,             # L2
        learning_rate="adaptive",
        max_iter=400,           # με early stopping
        early_stopping=False,
        n_iter_no_change=20,
        validation_fraction=0.1,
        batch_size="auto",
        random_state=42,
        verbose=False,
    )
    pipe = make_pipeline(vec, mlp)

    min_class_size = ytr.value_counts().min()
    n_splits = int(max(2, min(5, min_class_size)))
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipe, Xtr, ytr, scoring="f1_macro", cv=kf, n_jobs=-1)

    pipe.fit(Xtr, ytr)
    ypred = pipe.predict(Xte)

    report_dict = classification_report(yte, ypred, labels=labels_order, output_dict=True, zero_division=0)
    acc = accuracy_score(yte, ypred)
    prec, rec, f1, support = precision_recall_fscore_support(yte, ypred, labels=labels_order, zero_division=0)
    cm = confusion_matrix(yte, ypred, labels=labels_order)

    out_base = Path(OUTPUT_MLP) / f"mlp_{tag}"
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
            "cv_n_splits": n_splits,
        }, f, ensure_ascii=False, indent=2)

    pd.DataFrame(cm, index=labels_order, columns=labels_order).to_csv(out_base.with_suffix(".confusion.csv"), encoding="utf-8", index=True)

    summary_row = pd.DataFrame([{
        "scenario": tag,
        "model": "mlp_classifier",
        "accuracy": acc,
        "cv_macro_f1_mean": float(np.mean(cv_scores)),
        "cv_macro_f1_std": float(np.std(cv_scores)),
        "f1_macro": report_dict["macro avg"]["f1-score"],
        "precision_macro": report_dict["macro avg"]["precision"],
        "recall_macro": report_dict["macro avg"]["recall"],
        "f1_weighted": report_dict["weighted avg"]["f1-score"],
        "precision_weighted": report_dict["weighted avg"]["precision"],
        "recall_weighted": report_dict["weighted avg"]["recall"],
        "f1_micro": acc,
    }])
    summary_csv = Path(OUTPUT_MLP) / "mlp_summary_metrics.csv"
    if summary_csv.exists():
        prev = pd.read_csv(summary_csv)
        pd.concat([prev, summary_row], ignore_index=True).to_csv(summary_csv, index=False)
    else:
        summary_row.to_csv(summary_csv, index=False)

    joblib.dump(pipe, out_base.with_suffix(".joblib"))

    return {"pipe": pipe, "labels_order": labels_order, "cv_scores": cv_scores, "report": report_dict, "confusion": cm}

if __name__ == "__main__":
    df = load_data(INPUT)
    X_text = df["original_skill"]
    y_full = df["manual_label"]

    # ----- Binary
    y_bin, mask_bin = to_binary_labels(y_full)
    X_bin = X_text[mask_bin].reset_index(drop=True)
    y_bin = y_bin.reset_index(drop=True)
    labels_bin = sorted(y_bin.unique())
    train_eval_mlp(X_bin, y_bin, labels_bin, tag="binary_h0_vs_h1")

    # ----- Multiclass (≥5 δείγματα/κλάση)
    y_multi = y_full.copy()
    cls_counts = y_multi.value_counts()
    keep_classes = cls_counts[cls_counts >= 5].index.tolist()
    mask_multi = y_multi.isin(keep_classes)
    X_multi = X_text[mask_multi].reset_index(drop=True)
    y_multi = y_multi[mask_multi].reset_index(drop=True)
    labels_multi = sorted(y_multi.unique())

    if len(labels_multi) >= 2:
        train_eval_mlp(X_multi, y_multi, labels_multi, tag="multiclass_h0_h1subtypes")
    else:
        print("[INFO] Multiclass MLP skipped (not enough classes).")

    print("\n=== MLP baselines completed ===")
    print(f"Output folder: {OUTPUT_MLP}")
