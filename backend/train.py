import os
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import ComplementNB
import numpy as np
import re
import joblib

DEFAULT_DATASET = os.path.join(os.path.dirname(os.path.dirname(__file__)), "bharatfakenewskosh (3).xlsx")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")
MODEL_PATH = os.path.join(MODEL_DIR, "model.joblib")

TEXT_CANDIDATES = [
    "text", "content", "statement", "news", "headline", "title", "article",
]
LABEL_CANDIDATES = [
    "label", "target", "class", "verdict", "category",
]

def find_column(df, candidates):
    cols_lower = {c.lower(): c for c in df.columns}
    for name in candidates:
        if name in cols_lower:
            return cols_lower[name]
    # try fuzzy by containment
    for c in df.columns:
        cl = c.lower()
        for name in candidates:
            if name in cl or cl in name:
                return c
    return None


def build_pipeline(word_ngram=(1, 2), char_ngram=(3, 5), min_df=2, word_max_features=200_000,
                   char_max_features=200_000, solver="saga", max_iter=2000, C=1.0, random_state=42,
                   model_type="logreg", penalty="l2", l1_ratio=0.5):
    # Word n-grams capture semantics; char n-grams help with noisy Hindi/English text and misspellings.
    word_vec = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        analyzer="word",
        ngram_range=word_ngram,
        max_features=word_max_features,
        min_df=min_df,
    )
    char_vec = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        analyzer="char_wb",
        ngram_range=char_ngram,
        max_features=char_max_features,
        min_df=min_df,
    )
    features = FeatureUnion([
        ("word", word_vec),
        ("char", char_vec),
    ])
    if model_type == "logreg":
        # Elastic-net is supported only with saga
        if penalty == "elasticnet":
            clf = LogisticRegression(
                max_iter=max_iter,
                class_weight="balanced",
                solver="saga",
                n_jobs=-1,
                C=C,
                l1_ratio=l1_ratio,
                penalty="elasticnet",
                random_state=random_state,
            )
        else:
            clf = LogisticRegression(
                max_iter=max_iter,
                class_weight="balanced",
                solver=solver,
                n_jobs=-1,
                C=C,
                penalty="l2",
                random_state=random_state,
            )
    elif model_type == "linsvc":
        # Use calibrated SVM to obtain predict_proba for threshold tuning
        base = LinearSVC(C=C, class_weight="balanced", max_iter=max_iter, random_state=random_state)
        clf = CalibratedClassifierCV(estimator=base, cv=3)
    elif model_type == "cnb":
        clf = ComplementNB()
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")
    return Pipeline([
        ("feats", features),
        ("clf", clf),
    ])


_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_WS_RE = re.compile(r"\s+")
_EMOJI_RE = re.compile(r"[\U00010000-\U0010ffff]", flags=re.UNICODE)

def normalize_text(t: str) -> str:
    t = str(t)
    t = t.replace("\u200d", " ")  # zero-width joiner common in Indic text
    t = _URL_RE.sub(" ", t)
    t = _EMOJI_RE.sub(" ", t)
    t = t.strip().lower()
    t = _WS_RE.sub(" ", t)
    return t


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=DEFAULT_DATASET, help="Path to Excel/CSV dataset (ignored if --train_csv is set)")
    parser.add_argument("--sheet", default=0, help="Excel sheet index or name", type=lambda x: int(x) if str(x).isdigit() else x)
    # Optional precomputed splits
    parser.add_argument("--train_csv", default="", help="Path to train.csv with columns including Merged_Text and Label")
    parser.add_argument("--val_csv", default="", help="Path to val.csv with columns including Merged_Text and Label")
    parser.add_argument("--test_csv", default="", help="Path to test.csv with columns including Merged_Text and Label")
    # Colab-friendly controls
    parser.add_argument("--model_out", default=MODEL_PATH, help="Path to save trained model.joblib (e.g., /content/drive/MyDrive/model.joblib)")
    parser.add_argument("--cv", type=int, default=3, help="Cross-validation folds for GridSearchCV")
    parser.add_argument("--no_grid", action="store_true", help="Skip GridSearch and use default C")
    parser.add_argument("--C_grid", default="0.5,1.0,2.0", help="Comma-separated C values for GridSearchCV")
    parser.add_argument("--solver", default="saga", choices=["saga", "liblinear", "lbfgs"], help="LogReg solver (ignored for non-logreg)")
    parser.add_argument("--max_iter", type=int, default=2000, help="Max iterations for LogisticRegression")
    parser.add_argument("--word_max_features", type=int, default=200_000, help="TF-IDF max features for word analyzer")
    parser.add_argument("--char_max_features", type=int, default=200_000, help="TF-IDF max features for char analyzer")
    parser.add_argument("--min_df", type=int, default=2, help="Min document frequency for TF-IDF")
    parser.add_argument("--word_ngram", default="1,2", help="Word ngram range as start,end (e.g., 1,2)")
    parser.add_argument("--char_ngram", default="3,5", help="Char ngram range as start,end (e.g., 3,5)")
    parser.add_argument("--test_size", type=float, default=0.2, help="Test split size")
    parser.add_argument("--val_size", type=float, default=0.1, help="Validation split size from train for threshold tuning")
    parser.add_argument("--model_type", default="logreg", choices=["logreg","linsvc","cnb"], help="Classifier type")
    parser.add_argument("--penalty", default="l2", choices=["l2","elasticnet"], help="Penalty for logreg (elasticnet requires saga)")
    parser.add_argument("--l1_ratio_grid", default="0.2,0.5,0.8", help="Comma-separated l1_ratio values for elasticnet grid when model_type=logreg and penalty=elasticnet")
    args = parser.parse_args()

    use_splits = bool(args.train_csv)
    if use_splits:
        # Load pre-split CSVs
        if not (os.path.exists(args.train_csv) and os.path.exists(args.val_csv) and os.path.exists(args.test_csv)):
            raise FileNotFoundError("One or more split CSVs not found. Provide --train_csv, --val_csv, --test_csv")
        df_train = pd.read_csv(args.train_csv)
        df_val = pd.read_csv(args.val_csv)
        df_test = pd.read_csv(args.test_csv)
        # Prefer Merged_Text/Label if present
        def _pick_cols(df):
            tcol = "Merged_Text" if "Merged_Text" in df.columns else find_column(df, TEXT_CANDIDATES)
            lcol = "Label" if "Label" in df.columns else find_column(df, LABEL_CANDIDATES)
            if tcol is None or lcol is None:
                raise RuntimeError(f"Could not find text/label columns in split file. Columns={list(df.columns)}")
            return tcol, lcol
        tr_text_col, tr_label_col = _pick_cols(df_train)
        va_text_col, va_label_col = _pick_cols(df_val)
        te_text_col, te_label_col = _pick_cols(df_test)
        # Normalize and filter
        for dfx, tcol in [(df_train, tr_text_col), (df_val, va_text_col), (df_test, te_text_col)]:
            dfx[tcol] = dfx[tcol].astype(str).apply(normalize_text)
        X_tr, y_tr_raw = df_train[tr_text_col].tolist(), df_train[tr_label_col].astype(str).tolist()
        X_va, y_va_raw = df_val[va_text_col].tolist(), df_val[va_label_col].astype(str).tolist()
        X_te, y_te_raw = df_test[te_text_col].tolist(), df_test[te_label_col].astype(str).tolist()
        # Encode on train only
        le = LabelEncoder()
        y_tr = le.fit_transform(y_tr_raw)
        y_va = le.transform(y_va_raw)
        y_te = le.transform(y_te_raw)
        text_col = tr_text_col
        label_col = tr_label_col
    else:
        data_path = args.data
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Dataset not found: {data_path}")

        if data_path.lower().endswith((".xlsx", ".xls")):
            df = pd.read_excel(data_path, sheet_name=args.sheet, engine="openpyxl")
        else:
            df = pd.read_csv(data_path)

    if not use_splits:
        # Prefer explicit columns for Bharat Fake News Kosh schema
        preferred_text_cols = [
            "Eng_Trans_News_Body",
            "Eng_Trans_Statement",
            "News Body",
            "Statement",
            "Text",
        ]
        preferred_label_cols = [
            "Label",
            "label",
            "target",
        ]

        text_col = next((c for c in preferred_text_cols if c in df.columns), None)
        label_col = next((c for c in preferred_label_cols if c in df.columns), None)
        if text_col is None:
            text_col = find_column(df, TEXT_CANDIDATES)
        if label_col is None:
            label_col = find_column(df, LABEL_CANDIDATES)

        if text_col is None or label_col is None:
            raise RuntimeError(
                f"Could not detect text/label columns. Columns={list(df.columns)}\n"
                f"Tried text candidates={TEXT_CANDIDATES}, label candidates={LABEL_CANDIDATES}"
            )

        # Build a unified text from multiple possible columns
        concat_cols = [
            c for c in [
                "Eng_Trans_Statement",
                "Eng_Trans_News_Body",
                "Statement",
                "News Body",
                "Text",
            ] if c in df.columns
        ]
        if concat_cols:
            df["__concat_text__"] = df[concat_cols].astype(str).agg(" ".join, axis=1)
            text_col = "__concat_text__"

    if not use_splits:
        df = df[[text_col, label_col]].dropna().drop_duplicates()
        before = len(df)
        df[text_col] = df[text_col].astype(str).apply(normalize_text)
        # Filter short texts; if dataset collapses, relax the threshold
        df1 = df[df[text_col].str.len() >= 16]
        if len(df1) < 200:  # too small, relax
            df1 = df[df[text_col].str.len() >= 8]
        if len(df1) == 0:  # final safeguard
            df1 = df
        df = df1
        print(f"Rows before cleaning: {before}, after cleaning: {len(df)}")

        X = df[text_col].astype(str).tolist()
        y_raw = df[label_col].astype(str).tolist()

        le = LabelEncoder()
        y = le.fit_transform(y_raw)

        classes, counts = np.unique(y, return_counts=True)
        print("Label distribution (encoded -> count):", {int(c): int(n) for c, n in zip(classes, counts)})
        print("Label names (index -> name):", {i: name for i, name in enumerate(le.classes_)})

        # If dataset is very small, reduce test size to keep training stable; or use user-provided value
        test_size = args.test_size if len(df) >= 100 else (0.15 if len(df) >= 40 else args.test_size)
        strat = y if len(set(y)) > 1 else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=strat
        )
        # Hold out validation from train
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_train,
            test_size=args.val_size if len(X_train) > 0 else 0.1,
            random_state=42,
            stratify=y_train if len(set(y_train)) > 1 else None,
        )
    else:
        # Use provided splits
        X_tr, y_tr = X_tr, y_tr
        X_val, y_val = X_va, y_va
        X_test, y_test = X_te, y_te
        classes, counts = np.unique(y_tr, return_counts=True)
        print("Train label distribution (encoded -> count):", {int(c): int(n) for c, n in zip(classes, counts)})
        print("Label names (index -> name):", {i: name for i, name in enumerate(le.classes_)})

    # Parse ngram ranges
    def _parse_range(s):
        a, b = s.split(",")
        return (int(a), int(b))

    word_ngram = _parse_range(args.word_ngram)
    char_ngram = _parse_range(args.char_ngram)

    base_pipe = build_pipeline(
        word_ngram=word_ngram,
        char_ngram=char_ngram,
        min_df=args.min_df,
        word_max_features=args.word_max_features,
        char_max_features=args.char_max_features,
        solver=args.solver,
        max_iter=args.max_iter,
        C=1.0,
        model_type=args.model_type,
        penalty=args.penalty,
        l1_ratio=0.5,
        random_state=42,
    )

    # Quick hyperparameter search over C to improve generalization (can be skipped for speed)
    # Validation is already created above per branch

    if not args.no_grid:
        C_values = [float(x) for x in args.C_grid.split(",")]
        param_grid = {}
        if args.model_type == "logreg":
            if args.penalty == "elasticnet":
                l1_vals = [float(x) for x in args.l1_ratio_grid.split(",")]
                param_grid = {"clf__C": C_values, "clf__l1_ratio": l1_vals}
            else:
                param_grid = {"clf__C": C_values}
        elif args.model_type == "linsvc":
            # CalibratedClassifierCV param grid should target the wrapped estimator
            param_grid = {"clf__estimator__C": C_values}
        elif args.model_type == "cnb":
            # CNB has alpha smoothing; scan a few
            param_grid = {"clf__alpha": [0.5, 1.0, 2.0]}
        search = GridSearchCV(base_pipe, param_grid=param_grid, cv=args.cv, n_jobs=-1, verbose=1)
        search.fit(X_tr, y_tr)
        print("Best params:", search.best_params_)
        pipe = search.best_estimator_
    else:
        print("Skipping GridSearchCV; using default C=1.0")
        pipe = base_pipe.fit(X_tr, y_tr)

    # Threshold tuning for binary classification (if predict_proba available)
    tuned_threshold = None
    pos_index = None
    if hasattr(pipe, "predict_proba") and len(le.classes_) == 2:
        proba_val = pipe.predict_proba(X_val)
        # choose positive class index as the class with label name 'True' if present, else 1
        if "True" in le.classes_:
            pos_index = int(list(le.classes_).index("True"))
        else:
            pos_index = 1
        y_val_bin = (np.array(y_val) == pos_index).astype(int)
        best_f1 = -1.0
        best_th = 0.5
        for th in np.linspace(0.2, 0.8, 25):
            preds = (proba_val[:, pos_index] >= th).astype(int)
            # map back to multiclass labels 0/1
            # compute macro F1 manually
            from sklearn.metrics import f1_score
            f1 = f1_score(y_val_bin, preds, average='macro')
            if f1 > best_f1:
                best_f1 = f1
                best_th = float(th)
        tuned_threshold = best_th
        print(f"Tuned threshold (pos_index={pos_index}): {tuned_threshold:.3f}, val macro F1: {best_f1:.4f}")

    # Evaluate on test with tuned threshold if available
    if tuned_threshold is not None:
        proba_test = pipe.predict_proba(X_test)
        y_test_bin = (np.array(y_test) == pos_index).astype(int)
        test_preds_bin = (proba_test[:, pos_index] >= tuned_threshold).astype(int)
        # convert to label indices
        y_pred_idx = np.where(test_preds_bin==1, pos_index, 1-pos_index)
        acc = accuracy_score(y_test, y_pred_idx)
        print(f"Accuracy (tuned): {acc:.4f}")
        print(classification_report(y_test, y_pred_idx, target_names=list(le.classes_), zero_division=0))
    else:
        y_pred = pipe.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {acc:.4f}")
        print(classification_report(y_test, y_pred, target_names=list(le.classes_), zero_division=0))

    # Save model to the requested path (works with Google Drive paths on Colab)
    out_path = args.model_out
    out_dir = os.path.dirname(out_path)
    os.makedirs(out_dir, exist_ok=True)
    joblib.dump({
        "pipeline": pipe,
        "label_names": list(le.classes_),
        "text_col": text_col,
        "label_col": label_col,
        "threshold": tuned_threshold,
        "positive_index": pos_index,
    }, out_path)
    print(f"Saved model to {out_path}")


if __name__ == "__main__":
    main()
