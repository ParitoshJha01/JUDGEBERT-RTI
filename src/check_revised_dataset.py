from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "dataset" / "RTI_Revised_300_Pairs_Two_Assessors_Third_Adjudication.csv"
REQUIRED = ["Pair_ID", "Language", "Original_Text", "Simplified_Text", "LMP_Score", "Split"]

def main():
    if not DATA.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA}")
    df = pd.read_csv(DATA)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    if len(df) != 300:
        raise ValueError(f"Expected 300 rows; found {len(df)}")
    if df["Pair_ID"].isna().any() or not df["Pair_ID"].is_unique:
        raise ValueError("Pair_ID values must be present and unique.")
    if set(df["Language"].dropna().unique()) != {"English", "Hindi"}:
        raise ValueError("Expected English and Hindi only.")
    if df[["Original_Text", "Simplified_Text"]].isna().any().any():
        raise ValueError("Missing sentence text.")
    for c in ["Original_Text", "Simplified_Text"]:
        df[c] = df[c].astype(str).str.strip()
        if df[c].eq("").any():
            raise ValueError(f"Blank values in {c}.")
    df["LMP_Score"] = pd.to_numeric(df["LMP_Score"], errors="coerce")
    if df["LMP_Score"].isna().any() or not df["LMP_Score"].between(1, 10).all():
        raise ValueError("LMP_Score must be numeric and between 1 and 10.")
    expected_splits = {"train": 210, "validation": 44, "test": 46}
    actual = df["Split"].value_counts().to_dict()
    if actual != expected_splits:
        raise ValueError(f"Unexpected split sizes: {actual}")
    if not df.groupby("Language")["Split"].apply(lambda s: s.value_counts().to_dict() == {"train":105,"validation":22,"test":23}).all():
        raise ValueError("Unexpected language/split counts.")
    if "Provisional_Label" in df.columns:
        expected = df["LMP_Score"].map(
            lambda s: "Not Preserved" if s < 6 else ("Partially Preserved" if s < 8 else "Preserved")
        )
        bad = int((expected != df["Provisional_Label"]).sum())
        if bad:
            raise ValueError(f"{bad} Provisional_Label values do not match documented thresholds.")
    print("REVISED DATASET CHECK PASSED")
    print(f"Rows: {len(df)} | Unique Pair_IDs: {df.Pair_ID.nunique()}")
    print(df.groupby(["Language", "Split"]).size().to_string())
    print("\nLMP score summary:")
    print(df["LMP_Score"].describe().round(3).to_string())
    print("\nLabel counts:")
    if "Provisional_Label" in df.columns:
        print(df["Provisional_Label"].value_counts().to_string())
    print("\nReminder: use LMP_Score as the regression target, not Provisional_Label.")

if __name__ == "__main__":
    main()
