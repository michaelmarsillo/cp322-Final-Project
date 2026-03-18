import os
import pandas as pd
import numpy as np


COLUMN_NAMES = [
    "age",       # age in years
    "sex",       # 1 = male, 0 = female
    "cp",        # chest pain type (1-4)
    "trestbps",  # resting blood pressure (mm Hg)
    "chol",      # serum cholesterol (mg/dl)
    "fbs",       # fasting blood sugar > 120 mg/dl (1=true, 0=false)
    "restecg",   # resting ECG results (0, 1, 2)
    "thalach",   # maximum heart rate achieved
    "exang",     # exercise induced angina (1=yes, 0=no)
    "oldpeak",   # ST depression induced by exercise
    "slope",     # slope of peak exercise ST segment (1-3)
    "ca",        # number of major vessels colored by fluoroscopy (0-3)
    "thal",      # thal: 3=normal, 6=fixed defect, 7=reversable defect
    "target",    # diagnosis (0 = no disease, 1-4 = disease)
]

RAW_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
PROCESSED_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")


class DataLoader:
    """Loads and combines all 4 UCI Heart Disease datasets into one DataFrame."""

    def __init__(self, data_dir: str = RAW_DATA_DIR):
        self.data_dir = data_dir
        self.files = [
            "processed.cleveland.data",
            "processed.hungarian.data",
            "processed.switzerland.data",
            "processed.va.data",
        ]

    def load(self) -> pd.DataFrame:
        frames = []
        for fname in self.files:
            path = os.path.join(self.data_dir, fname)
            df = pd.read_csv(
                path,
                header=None,
                names=COLUMN_NAMES,
                na_values="?",
            )
            source = fname.replace("processed.", "").replace(".data", "")
            df["source"] = source
            frames.append(df)
            print(f"  Loaded {fname}: {len(df)} rows")

        combined = pd.concat(frames, ignore_index=True)
        print(f"\nTotal combined rows: {len(combined)}")
        return combined


class DataPreprocessor:
    """Cleans and prepares the heart disease dataset for modeling."""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def binarize_target(self):
        """Convert target from 0-4 scale to binary: 0=no disease, 1=disease."""
        self.df["target"] = (self.df["target"] > 0).astype(int)
        return self

    def drop_source_column(self):
        """Remove the source tracking column before modeling."""
        if "source" in self.df.columns:
            self.df = self.df.drop(columns=["source"])
        return self

    def report_missing(self):
        """Print a summary of missing values per column."""
        missing = self.df.isnull().sum()
        missing_pct = (missing / len(self.df) * 100).round(2)
        report = pd.DataFrame({"missing_count": missing, "missing_pct": missing_pct})
        report = report[report["missing_count"] > 0].sort_values("missing_pct", ascending=False)
        print("\nMissing value report:")
        print(report.to_string())
        return self

    def impute_missing(self):
        """
        Impute missing values:
        - Numerical columns: median imputation (robust to outliers)
        - Categorical columns: mode imputation
        """
        categorical_cols = ["cp", "restecg", "slope", "thal", "ca", "fbs", "exang", "sex"]
        numerical_cols = ["age", "trestbps", "chol", "thalach", "oldpeak"]

        for col in numerical_cols:
            if col in self.df.columns and self.df[col].isnull().any():
                median_val = self.df[col].median()
                self.df[col] = self.df[col].fillna(median_val)
                print(f"  Imputed '{col}' with median: {median_val}")

        for col in categorical_cols:
            if col in self.df.columns and self.df[col].isnull().any():
                mode_val = self.df[col].mode()[0]
                self.df[col] = self.df[col].fillna(mode_val)
                print(f"  Imputed '{col}' with mode: {mode_val}")

        return self

    def remove_outliers(self):
        """
        Remove extreme outliers using IQR method on numerical columns.
        Rows where any numerical feature is beyond 3*IQR from Q1/Q3 are dropped.
        """
        numerical_cols = ["trestbps", "chol", "thalach", "oldpeak"]
        initial_len = len(self.df)

        for col in numerical_cols:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 3 * IQR
            upper = Q3 + 3 * IQR
            self.df = self.df[(self.df[col] >= lower) & (self.df[col] <= upper)]

        removed = initial_len - len(self.df)
        print(f"\nOutlier removal: dropped {removed} rows (kept {len(self.df)})")
        return self

    def get_dataframe(self) -> pd.DataFrame:
        return self.df.reset_index(drop=True)

    def save(self, filename: str = "heart_disease_clean.csv"):
        os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
        out_path = os.path.join(PROCESSED_DATA_DIR, filename)
        self.df.to_csv(out_path, index=False)
        print(f"\nSaved cleaned data to: {out_path}")
        return out_path


def run_preprocessing(save: bool = True) -> pd.DataFrame:
    """
    Full preprocessing pipeline: load -> binarize -> impute -> remove outliers -> save.
    Returns the cleaned DataFrame.
    """
    print("=" * 50)
    print("Step 1: Loading raw data")
    print("=" * 50)
    loader = DataLoader()
    df_raw = loader.load()

    print("\n" + "=" * 50)
    print("Step 2: Preprocessing")
    print("=" * 50)
    preprocessor = (
        DataPreprocessor(df_raw)
        .binarize_target()
        .drop_source_column()
        .report_missing()
        .impute_missing()
        .remove_outliers()
    )

    df_clean = preprocessor.get_dataframe()

    print(f"\nFinal dataset shape: {df_clean.shape}")
    print(f"Class distribution:\n{df_clean['target'].value_counts().to_string()}")

    if save:
        preprocessor.save()

    return df_clean


if __name__ == "__main__":
    run_preprocessing()
