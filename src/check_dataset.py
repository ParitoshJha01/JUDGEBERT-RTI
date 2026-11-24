import pandas as pd
from pathlib import Path

# Run this script from the main "Judgeberrt RTI" project folder.
DATA_PATH = Path("dataset/RTI_Model_Ready_Dataset_Grouped_Split.csv")

print("=" * 55)
print("JUDGEBERT RTI - DATASET VERIFICATION")
print("=" * 55)

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}\n"
        "Make sure the CSV is inside the project's dataset folder."
    )

df = pd.read_csv(DATA_PATH)

required_columns = [
    "Pair_ID",
    "Language",
    "Original_Text",
    "Simplified_Text",
    "LMP_Score",
    "Provisional_Label",
    "Split",
    "Source_Group",
]

print("\n1. Dataset shape:", df.shape)

missing_columns = [col for col in required_columns if col not in df.columns]
if missing_columns:
    raise ValueError(f"Missing required columns: {missing_columns}")
print("2. Required columns: PASS")

print("\n3. Missing values:")
print(df[required_columns].isnull().sum())

print("\n4. Language distribution:")
print(df["Language"].value_counts(dropna=False))

print("\n5. Train/validation/test distribution:")
print(pd.crosstab(df["Language"], df["Split"], dropna=False))

print("\n6. LMP score statistics:")
print(df["LMP_Score"].describe())

scores = pd.to_numeric(df["LMP_Score"], errors="coerce")
print("\n7. Invalid or missing scores:", int((~scores.between(1, 10)).sum()))

empty_text = (
    df["Original_Text"].fillna("").astype(str).str.strip().eq("")
    | df["Simplified_Text"].fillna("").astype(str).str.strip().eq("")
)
print("8. Empty text pairs:", int(empty_text.sum()))

print("9. Duplicate Pair IDs:", int(df["Pair_ID"].duplicated().sum()))

group_split_counts = df.groupby("Source_Group")["Split"].nunique()
print(
    "10. Source groups crossing splits:",
    int((group_split_counts > 1).sum())
)

print("\nDATASET VERIFICATION COMPLETED.")
