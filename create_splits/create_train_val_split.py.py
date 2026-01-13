import pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit

root = Path(r"G:/My Drive/hindi_dfake")
meta = root / "metadata"
in_csv = meta / "fs_trainval_rest.ptm.csv"

# Read the big train+val pool
df = pd.read_csv(in_csv)

# Use speaker_id for group split if present and non-empty
if "speaker_id" in df.columns and df["speaker_id"].notna().any():
    speakers = df["speaker_id"].fillna("unk")
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.10, random_state=42)
    train_idx, val_idx = next(splitter.split(df, groups=speakers))
    train_df = df.iloc[train_idx].copy()
    val_df   = df.iloc[val_idx].copy()
else:
    # fallback: plain random split
    val_df = df.sample(frac=0.10, random_state=42)
    train_df = df.drop(val_df.index)

# Write outputs
out_train = meta / "split_train.ptm.csv"
out_val   = meta / "split_val.ptm.csv"

train_df.to_csv(out_train, index=False)
val_df.to_csv(out_val, index=False)

print(f"Train: {len(train_df):,}  Val: {len(val_df):,}")
print(f"Wrote:\n  {out_train}\n  {out_val}")
