import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # project root (LOBI-Backtest)
TRAIN_PATH = ROOT / "data" / "FI2010_train.csv"
TEST_PATH = ROOT / "data" / "FI2010_test.csv"


def inspect(path: str, name: str) -> None:
    print(f"\n================ {name} ================")
    df = pd.read_csv(path)
    print("shape:", df.shape)
    print("\ncolumns:")
    print(list(df.columns))
    print("\nhead:")
    print(df.head())
    print("\ndtypes:")
    print(df.dtypes)
    print("\ndescribe (numeric):")
    print(df.describe())
    print("\nmissing values (top 20):")
    print(df.isna().sum().sort_values(ascending=False).head(20))
    # Identify columns with small number of unique values (likely labels/classes)
    nunique = df.nunique()
    candidate = nunique[nunique <= 10].sort_values()
    print("\n--- columns with <=10 unique values (top 30) ---")
    print(candidate.head(30))

    label_cols = ["144", "145", "146", "147", "148"]
    print("\n--- label value counts (TRAIN) ---")
    for c in label_cols:
        if c in df.columns:
            print(f"\ncol {c}:")
            print(df[c].value_counts(dropna=False).sort_index())


if __name__ == "__main__":
    inspect(TRAIN_PATH, "TRAIN")
    inspect(TEST_PATH, "TEST")
