import os
import pandas as pd
import re
import numpy as np

def unify_label(lbl):
    if lbl is None:
        return "unknown"
    s = str(lbl).strip().lower()

    # simple cases
    if s in ["fake", "false", "0", "f"]:
        return "fake"
    if s in ["real", "true", "1", "t"]:
        return "real"

    return "unknown"


rows = []

# ----------------------------------------------------------------------
# 1) BharatKosh (Hindi)
# ----------------------------------------------------------------------

bk_path = "data/bharatfakenewskosh (3).xlsx"
if os.path.exists(bk_path):
    df = pd.read_excel(bk_path)

    text_col = "Statement"
    body_col = "News Body"
    label_col = "Label"

    for _, r in df.iterrows():
        merged = str(r.get(text_col, "")) + " " + str(r.get(body_col, ""))
        
        rows.append({
            "source": "bharatkosh",
            "Title": "",
            "Text": merged,
            "Merged_Text": merged,
            "orig_label": r.get(label_col, None),
            "language": "hi"
        })


# ----------------------------------------------------------------------
# 2) Hindi FNR (Hindi folder of .txt files) with light cleaning
# ----------------------------------------------------------------------

def load_txt_dir(path, label):
    if not os.path.exists(path):
        return
    for fname in os.listdir(path):
        if fname.endswith(".txt"):
            fpath = os.path.join(path, fname)
            txt = open(fpath, "r", encoding="utf-8", errors="ignore").read()
            # basic cleanup: remove URLs, collapse whitespace, drop obvious boilerplate tokens
            txt = re.sub(r"https?://\S+", " ", txt)
            txt = re.sub(r"\s+", " ", txt).strip()
            # remove lines that are pure disclaimers or repeated boilerplate markers
            boilerplate_keywords = [
                "डिसक्लेमर", "disclaimer", "यह भी पढ़ें", "यह भी पढ़ें", "यह भी पढ़े",
            ]
            lowered = txt.lower()
            for kw in boilerplate_keywords:
                lowered = lowered.replace(kw.lower(), " ")
            txt = re.sub(r"\s+", " ", lowered).strip()
            rows.append({
                "source": "hindi_fnr",
                "Title": "",
                "Text": txt,
                "Merged_Text": txt,
                "orig_label": label,
                "language": "hi"
            })

load_txt_dir("data/Hindi_F&R_News/Hindi_fake_news", "fake")
load_txt_dir("data/Hindi_F&R_News/Hindi_real_news", "real")


# ----------------------------------------------------------------------
# 3) ISOT (English)
# ----------------------------------------------------------------------

if os.path.exists("data/ISOT/True.csv"):
    df = pd.read_csv("data/ISOT/True.csv")
    for _, r in df.iterrows():
        title = r.get("title", "")
        text = r.get("text", "")
        merged = f"{title} {text}".strip()
        rows.append({
            "source": "isot",
            "Title": title,
            "Text": text,
            "Merged_Text": merged,
            "orig_label": "real",
            "language": "en"
        })

if os.path.exists("data/ISOT/Fake.csv"):
    df = pd.read_csv("data/ISOT/Fake.csv")
    for _, r in df.iterrows():
        title = r.get("title", "")
        text = r.get("text", "")
        merged = f"{title} {text}".strip()
        rows.append({
            "source": "isot",
            "Title": title,
            "Text": text,
            "Merged_Text": merged,
            "orig_label": "fake",
            "language": "en"
        })


# ----------------------------------------------------------------------
# 4) GossipCop (English, prefer title + content)
# ----------------------------------------------------------------------

if os.path.exists("data/gossipcop/gossipcop_fake.csv"):
    df = pd.read_csv("data/gossipcop/gossipcop_fake.csv")
    for _, r in df.iterrows():
        title = str(r.get("title", ""))
        content = str(r.get("content", r.get("text", "")))
        merged = f"{title} {content}".strip() if (title or content) else title
        txt = merged or title
        rows.append({
            "source": "gossipcop",
            "Title": title,
            "Text": content or title,
            "Merged_Text": txt,
            "orig_label": "fake",
            "language": "en"
        })

if os.path.exists("data/gossipcop/gossipcop_real.csv"):
    df = pd.read_csv("data/gossipcop/gossipcop_real.csv")
    for _, r in df.iterrows():
        title = str(r.get("title", ""))
        content = str(r.get("content", r.get("text", "")))
        merged = f"{title} {content}".strip() if (title or content) else title
        txt = merged or title
        rows.append({
            "source": "gossipcop",
            "Title": title,
            "Text": content or title,
            "Merged_Text": txt,
            "orig_label": "real",
            "language": "en"
        })


# ----------------------------------------------------------------------
# 5) PolitiFact (English, prefer title + content)
# ----------------------------------------------------------------------

if os.path.exists("data/politifact/politifact_fake.csv"):
    df = pd.read_csv("data/politifact/politifact_fake.csv")
    for _, r in df.iterrows():
        title = str(r.get("title", ""))
        content = str(r.get("content", r.get("text", "")))
        merged = f"{title} {content}".strip() if (title or content) else title
        txt = merged or title
        rows.append({
            "source": "politifact",
            "Title": title,
            "Text": content or title,
            "Merged_Text": txt,
            "orig_label": "fake",
            "language": "en"
        })

if os.path.exists("data/politifact/politifact_real.csv"):
    df = pd.read_csv("data/politifact/politifact_real.csv")
    for _, r in df.iterrows():
        title = str(r.get("title", ""))
        content = str(r.get("content", r.get("text", "")))
        merged = f"{title} {content}".strip() if (title or content) else title
        txt = merged or title
        rows.append({
            "source": "politifact",
            "Title": title,
            "Text": content or title,
            "Merged_Text": txt,
            "orig_label": "real",
            "language": "en"
        })


# ----------------------------------------------------------------------
# FINAL CLEANING
# ----------------------------------------------------------------------

df = pd.DataFrame(rows)
df["Label"] = df["orig_label"].apply(unify_label)
df = df[df["Label"] != "unknown"]

df = df[df["Merged_Text"].str.len() > 10]

df["Statement"] = df.apply(lambda r: r["Merged_Text"] if r["language"] == "hi" else "", axis=1)
df["Eng_Trans_Statement"] = df.apply(lambda r: r["Merged_Text"] if r["language"] == "en" else "", axis=1)

# Normalize whitespace and drop duplicates
df["Merged_Text"] = df["Merged_Text"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
df = df.drop_duplicates(subset=["Merged_Text", "Label"]).reset_index(drop=True)

out_path = "data/bilingual_merged.xlsx"
df[["Title","Text","Statement","Eng_Trans_Statement","language","Label","source"]].to_excel(out_path, index=False)

print("Saved:", out_path)
print("Total final samples:", len(df))

# Class distribution and splits
print("Class distribution:\n", df["Label"].value_counts())

def stratified_split(df_in, label_col, ratios=(0.8, 0.1, 0.1), seed=42):
    assert abs(sum(ratios) - 1.0) < 1e-6
    np.random.seed(seed)
    parts = []
    for lbl, g in df_in.groupby(label_col):
        idx = np.random.permutation(len(g))
        n = len(g)
        n_train = int(ratios[0] * n)
        n_val = int(ratios[1] * n)
        tr = g.iloc[idx[:n_train]]
        va = g.iloc[idx[n_train:n_train+n_val]]
        te = g.iloc[idx[n_train+n_val:]]
        parts.append((tr, va, te))
    train = pd.concat([p[0] for p in parts]).sample(frac=1, random_state=seed).reset_index(drop=True)
    val = pd.concat([p[1] for p in parts]).sample(frac=1, random_state=seed).reset_index(drop=True)
    test = pd.concat([p[2] for p in parts]).sample(frac=1, random_state=seed).reset_index(drop=True)
    return train, val, test

train_df, val_df, test_df = stratified_split(df, "Label", ratios=(0.8,0.1,0.1), seed=42)

os.makedirs("data/splits", exist_ok=True)
train_path = "data/splits/train.csv"
val_path = "data/splits/val.csv"
test_path = "data/splits/test.csv"
train_df.to_csv(train_path, index=False)
val_df.to_csv(val_path, index=False)
test_df.to_csv(test_path, index=False)
print("Saved splits:", train_path, val_path, test_path)

# Optional Parquet export if available
try:
    parquet_path = "data/bilingual_merged.parquet"
    df_parquet = df.copy()
    # Ensure stable dtypes for Parquet
    if "orig_label" in df_parquet.columns:
        df_parquet["orig_label"] = df_parquet["orig_label"].astype(str)
    df_parquet["Label"] = df_parquet["Label"].astype(str)
    df_parquet["language"] = df_parquet["language"].astype(str)
    df_parquet.to_parquet(parquet_path, index=False)
    print("Saved Parquet:", parquet_path)
except Exception as e:
    print("Parquet not saved:", e)
