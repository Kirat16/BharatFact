import os
import argparse
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from langdetect import detect, DetectorFactory

# make langdetect deterministic
DetectorFactory.seed = 42

# Defaults
MODEL_NAME = os.environ.get("INDIC_MODEL", "google/muril-base-cased")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "transformer_model")
DEFAULT_DATASET = os.path.join(os.path.dirname(os.path.dirname(__file__)), "bharatfakenewskosh (3).xlsx")

TEXT_PREFS = [
    # Prioritize English translations first
    "Eng_Trans_Statement",
    "Eng_Trans_News_Body",
    # Common English headline/title fields
    "Headline",
    "Title",
    # Hindi originals (fallbacks)
    "Statement",
    "News Body",
    "Hin_Statement",
    "Hin_News_Body",
    "Hindi Statement",
    "Hindi News Body",
    # Generic catch-all
    "Text",
]
LABEL_PREFS = ["Label", "label", "target", "class", "verdict", "category"]


def find_col(df: pd.DataFrame, prefs: List[str]):
    for c in prefs:
        if c in df.columns:
            return c
    # fallback by lowercase contains
    low = {c.lower(): c for c in df.columns}
    for p in prefs:
        if p.lower() in low:
            return low[p.lower()]
    return None


def build_text(df: pd.DataFrame, headline_weight: int = 3) -> pd.DataFrame:
    cols = [c for c in TEXT_PREFS if c in df.columns]
    if cols:
        df = df.copy()
        # Headline/Title weighting if present
        headline_cols = [c for c in ["Headline", "Title"] if c in df.columns]
        other_cols = [c for c in cols if c not in headline_cols]
        pieces = []
        if headline_cols:
            # duplicate headline twice for extra weight
            for _ in range(max(1, int(headline_weight))):
                pieces.append(df[headline_cols].astype(str).agg(" ".join, axis=1))
        if other_cols:
            pieces.append(df[other_cols].astype(str).agg(" ".join, axis=1))
        if not pieces:
            pieces = [pd.Series([""] * len(df))]
        df["__text__"] = pd.concat(pieces, axis=1).astype(str).agg(" ".join, axis=1)
        return df, "__text__"
    c = find_col(df, TEXT_PREFS)
    if c is None:
        raise RuntimeError(f"Could not find text columns, columns={list(df.columns)}")
    return df, c


def detect_lang_safe(text: str) -> str:
    try:
        return detect(text)
    except Exception:
        return "unk"


def normalize_lang(value: str) -> str:
    if value is None:
        return "unk"
    v = str(value).strip().lower()
    if v in {"hindi", "hin", "hi-in", "hi"}:
        return "hi"
    if v in {"english", "eng", "en-in", "en", "us", "uk"}:
        return "en"
    # common two-letter fallbacks
    if len(v) >= 2:
        return v[:2]
    return v or "unk"


class NewsDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], tokenizer: AutoTokenizer, max_length: int = 256):
        self.texts = texts
        self.labels = labels
        self.tok = tokenizer
        self.max_len = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        t = str(self.texts[idx])
        enc = self.tok(
            t,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
        )
        enc["labels"] = int(self.labels[idx])
        return {k: torch.tensor(v) for k, v in enc.items()}


@dataclass
class ComputeMetrics:
    label_names: List[str]

    def __call__(self, eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        acc = accuracy_score(labels, preds)
        f1_macro = f1_score(labels, preds, average="macro")
        return {"accuracy": acc, "f1_macro": f1_macro}


class WeightedTrainer(Trainer):
    def __init__(self, *args, class_weights=None, use_focal=False, focal_gamma=2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        if self.class_weights is not None:
            self.class_weights = torch.tensor(self.class_weights, dtype=torch.float)
        self.use_focal = use_focal
        self.focal_gamma = float(focal_gamma)

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**{k: v for k, v in inputs.items() if k != "labels"})
        logits = outputs.get("logits")
        if self.use_focal:
            # Focal loss implementation (binary/multi-class)
            ce = torch.nn.functional.cross_entropy(
                logits, labels, weight=(self.class_weights.to(logits.device) if self.class_weights is not None else None), reduction='none'
            )
            # pt = exp(-ce)
            pt = torch.exp(-ce)
            loss = ( (1 - pt) ** self.focal_gamma * ce ).mean()
        else:
            loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device) if self.class_weights is not None else None)
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

    def get_train_dataloader(self) -> DataLoader:
        # Weighted sampler to address class imbalance
        if self.train_dataset is None:
            return super().get_train_dataloader()
        labels = torch.tensor(getattr(self.train_dataset, "labels"))
        unique, counts = torch.unique(labels, return_counts=True)
        class_count = {int(u): int(c) for u, c in zip(unique.tolist(), counts.tolist())}
        # allow strengthening sampling via trainer.args (we'll store on args)
        gamma = getattr(self.args, 'sampler_gamma', 1.0)
        base_weights = torch.tensor([1.0 / class_count[int(y)] for y in labels.tolist()], dtype=torch.float)
        sample_weights = base_weights ** gamma
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(self.train_dataset), replacement=True)
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=sampler,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=DEFAULT_DATASET)
    parser.add_argument("--sheet", default=0, type=lambda x: int(x) if str(x).isdigit() else x)
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--min_text_len", type=int, default=5)
    parser.add_argument("--headline_weight", type=int, default=3, help="Times to repeat Headline/Title to emphasize")
    parser.add_argument("--use_focal", action="store_true", help="Use focal loss instead of cross-entropy")
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    parser.add_argument("--sampler_gamma", type=float, default=1.0, help=">1 strengthens minority upsampling in sampler")
    parser.add_argument("--pos_label", type=str, default="True", help="Label name to treat as positive class for threshold tuning (binary only)")
    # Language filtering for training: pass comma-separated codes like "hi,en"; empty = no filtering
    parser.add_argument("--langs", type=str, default="", help="Comma-separated language codes to keep during TRAINING (e.g., hi,en). Empty=keep all.")
    # Languages to include in EVALUATION only; training still uses all unless --langs is set
    parser.add_argument("--eval_langs", type=str, default="hi,en", help="Comma-separated language codes to include in eval (val/test). Empty=all.")
    args = parser.parse_args()

    if not os.path.exists(args.data):
        raise FileNotFoundError(args.data)

    df = pd.read_excel(args.data, sheet_name=args.sheet) if args.data.lower().endswith((".xlsx",".xls")) else pd.read_csv(args.data)
    df, text_col = build_text(df, headline_weight=args.headline_weight)
    label_col = find_col(df, LABEL_PREFS)
    if label_col is None:
        raise RuntimeError("Could not find label column")

    # Detect potential language column BEFORE reducing columns
    lang_candidates = [c for c in df.columns if c.lower() in {"language", "lang", "lng", "lang_code", "language_code"}]
    lang_col_name = lang_candidates[0] if lang_candidates else None

    # Keep text, label, and language (if present)
    cols_to_keep = [text_col, label_col] + ([lang_col_name] if lang_col_name else [])
    df = df[cols_to_keep].dropna().drop_duplicates()
    before_len = len(df)
    df = df[df[text_col].astype(str).str.len() >= args.min_text_len]
    print(f"Dropped {before_len - len(df)} rows below min_text_len={args.min_text_len}")

    # Language detection or dataset-provided language and filtering to Hindi/English
    print("Determining languages (prefer dataset column if available)...")
    if lang_col_name:
        df["__lang__"] = df[lang_col_name].map(normalize_lang)
        print(f"Using dataset language column: {lang_col_name}")
    else:
        print("No language column found; running langdetect (may be slow)...")
        df["__lang__"] = df[text_col].astype(str).map(detect_lang_safe).map(normalize_lang)
    lang_counts = df["__lang__"].value_counts(dropna=False).to_dict()
    print("Language distribution (before filter):", lang_counts)
    if args.langs.strip():
        keep_langs = {s.strip().lower() for s in args.langs.split(",") if s.strip()}
        df = df[df["__lang__"].isin(keep_langs)]
        print(f"Retained rows after language filter {keep_langs}: {len(df)}")
    else:
        print("No language filtering applied (keeping all languages).")

    # Prepend language token to text to help the model
    df[text_col] = "[lang=" + df["__lang__"] + "] " + df[text_col].astype(str)

    # Encode labels
    le = LabelEncoder()
    df["__y__"] = le.fit_transform(df[label_col].astype(str).tolist())

    # Optional TRAINING filter
    if args.langs.strip():
        keep_langs = {s.strip().lower() for s in args.langs.split(",") if s.strip()}
        df_trainable = df[df["__lang__"].isin(keep_langs)].copy()
        print(f"Training will use only languages {keep_langs}: rows={len(df_trainable)}")
    else:
        df_trainable = df
        print(f"Training uses all languages: rows={len(df_trainable)}")

    # Build stratify key on the TRAINABLE set
    strat_all = np.array([
        f"{y}_{l}" for y, l in zip(df_trainable["__y__"].tolist(), df_trainable["__lang__"].astype(str).tolist())
    ])
    idx_all = np.arange(len(df_trainable))

    # Primary split indices (stratified by label+lang)
    idx_train, idx_test, strat_train, strat_test = train_test_split(
        idx_all, strat_all, test_size=0.15, random_state=42, stratify=strat_all
    )
    # Secondary split for validation (stratified by train labels+lang)
    idx_train, idx_val, strat_train_final, strat_val = train_test_split(
        idx_train, strat_train, test_size=0.15, random_state=42, stratify=strat_train
    )

    # Map splits to the TRAINABLE dataframe
    df_tr = df_trainable.iloc[idx_train]
    df_va = df_trainable.iloc[idx_val]
    df_te = df_trainable.iloc[idx_test]

    # EVAL filtering: reduce val/test to specified languages (default hi,en)
    if args.eval_langs.strip():
        eval_keep = {s.strip().lower() for s in args.eval_langs.split(",") if s.strip()}
        df_va = df_va[df_va["__lang__"].isin(eval_keep)]
        df_te = df_te[df_te["__lang__"].isin(eval_keep)]
        print(f"Eval languages {eval_keep}: val={len(df_va)}, test={len(df_te)}")
    else:
        print(f"Eval uses all languages: val={len(df_va)}, test={len(df_te)}")

    # Build datasets
    X_train = df_tr[text_col].astype(str).tolist()
    y_train = df_tr["__y__"].astype(int).tolist()
    X_val = df_va[text_col].astype(str).tolist()
    y_val = df_va["__y__"].astype(int).tolist()
    X_test = df_te[text_col].astype(str).tolist()
    y_test = df_te["__y__"].astype(int).tolist()

    # Diagnostics: label distributions and sample texts
    from collections import Counter
    print("Train label distribution:", dict(Counter(y_train)))
    print("Val label distribution:", dict(Counter(y_val)))
    print("Test label distribution:", dict(Counter(y_test)))
    print("Sample training texts (first 5):")
    for i in range(min(5, len(X_train))):
        print(f"[{i}] y={y_train[i]} :: {X_train[i][:300]}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    # Add special language tokens so [lang=xx] is treated as a single token
    unique_langs = sorted(df["__lang__"].unique().tolist())
    add_tokens = [f"[lang={l}]" for l in unique_langs if l in ("hi", "en", "unk")]
    # Always ensure hi/en/unk are present
    for base_tok in ("[lang=hi]", "[lang=en]", "[lang=unk]"):
        if base_tok not in add_tokens:
            add_tokens.append(base_tok)
    special_tokens_dict = {"additional_special_tokens": add_tokens}
    num_added = tokenizer.add_special_tokens(special_tokens_dict)
    print(f"Added {num_added} special tokens to tokenizer: {add_tokens}")
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=len(le.classes_))
    if num_added > 0:
        model.resize_token_embeddings(len(tokenizer))
    # Verify tokenizer/model vocab alignment
    try:
        print(f"Model vocab size: {model.get_input_embeddings().weight.size(0)}")
        print(f"Tokenizer vocab size: {len(tokenizer)}")
    except Exception as e:
        print("Vocab size check error:", e)

    # class weights for imbalance (scikit-learn balanced)
    y_train_arr = np.array(y_train)
    classes = np.unique(y_train_arr)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train_arr)
    print("Calculated Weights (Scikit):", {int(c): float(w) for c, w in zip(classes, weights)})

    # Datasets
    train_ds = NewsDataset(X_train, y_train, tokenizer, args.max_len)
    val_ds = NewsDataset(X_val, y_val, tokenizer, args.max_len)
    test_ds = NewsDataset(X_test, y_test, tokenizer, args.max_len)

    metric_fn = ComputeMetrics(label_names=list(le.classes_))

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        logging_steps=50,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        report_to=[],
        fp16=torch.cuda.is_available(),
        # Tunings
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        gradient_accumulation_steps=2,
        label_smoothing_factor=args.label_smoothing,
        remove_unused_columns=False,
        # pass custom sampler strength through args for dataloader
        # (HF will ignore unknown args except those we defined)
        # We attach as attributes post-creation below.
    )

    # attach custom arg to trainer args for dataloader use
    setattr(training_args, 'sampler_gamma', args.sampler_gamma)

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=metric_fn,
        # TEMP: disable class weights to isolate issue
        class_weights=None,
        use_focal=args.use_focal,
        focal_gamma=args.focal_gamma,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=4)],
    )

    trainer.train()

    # Evaluate
    # Threshold tuning for binary tasks using validation set probabilities
    logits_val = trainer.predict(val_ds).predictions
    probs_val = torch.softmax(torch.tensor(logits_val), dim=-1).numpy()
    pos_idx = 1
    # try to find label named as args.pos_label
    try:
        pos_idx = int(list(le.classes_).index(str(args.pos_label)))
    except ValueError:
        pos_idx = 1 if len(le.classes_) >= 2 else 0
    best_th, best_f1 = 0.5, -1.0
    if len(le.classes_) == 2:
        y_val_bin = (np.array(y_val) == pos_idx).astype(int)
        for th in np.linspace(0.05, 0.95, 91):
            pred_bin = (probs_val[:, pos_idx] >= th).astype(int)
            f1 = f1_score(y_val_bin, pred_bin, average='macro')
            if f1 > best_f1:
                best_f1, best_th = float(f1), float(th)
        print(f"Tuned threshold on val: th={best_th:.3f}, f1_macro={best_f1:.4f}")

    # Evaluate on test using tuned threshold if binary, else argmax
    preds = trainer.predict(test_ds)
    logits_test = preds.predictions
    if len(le.classes_) == 2:
        probs_test = torch.softmax(torch.tensor(logits_test), dim=-1).numpy()
        y_pred = np.where(probs_test[:, pos_idx] >= best_th, pos_idx, 1 - pos_idx)
    else:
        y_pred = np.argmax(logits_test, axis=-1)
    acc = accuracy_score(y_test, y_pred)
    f1m = f1_score(y_test, y_pred, average="macro")
    print(f"Test Accuracy: {acc:.4f} | F1-macro: {f1m:.4f}")
    print(classification_report(y_test, y_pred, target_names=list(le.classes_)))
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion matrix (rows=true, cols=pred):")
    print(cm)
    try:
        pd.DataFrame(cm, index=list(le.classes_), columns=list(le.classes_)).to_csv(os.path.join(OUTPUT_DIR, "confusion_matrix.csv"), index=True)
    except Exception:
        pass

    # Per-language confusion (if available)
    try:
        langs_test = df_te["__lang__"].astype(str).tolist()
        unique_langs = sorted(set(langs_test))
        for lg in unique_langs:
            mask = [l == lg for l in langs_test]
            if any(mask):
                cm_l = confusion_matrix(np.array(y_test)[mask], np.array(y_pred)[mask])
                print(f"Confusion matrix for lang={lg} (rows=true, cols=pred):")
                print(cm_l)
    except Exception:
        pass

    # Save model and tokenizer + label names
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    with open(os.path.join(OUTPUT_DIR, "labels.txt"), "w", encoding="utf-8") as f:
        for name in le.classes_:
            f.write(str(name) + "\n")
    print(f"Saved transformer model to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
