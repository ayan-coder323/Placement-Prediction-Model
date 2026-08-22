from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


DATASET_PATH = Path(__file__).resolve().parent.parent / "dataset" / "placement_predict_50K_Raw.csv"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"


def main() -> None:
 print("1. Load the Dataset")

 try:
        df = pd.read_csv(DATASET_PATH)

        print("-----------------------------------")
        print("1. Dataset Contents:")
        print("-----------------------------------")
        print(df)

        print("-----------------------------------")
        print("\n2. Number of Rows and Columns:", df.shape)
        print("-----------------------------------")

        print("\n3. Column Names:")
        print("-----------------------------------")
        print(df.columns.tolist())

        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 1000)

        print("-----------------------------------")
        print("\n4. --- Placement Predict CSV Dataset Table View ---")
        print("-----------------------------------")
        print("Dataset first 10 records")
        print("-----------------------------------")
        print(df.head(10))
        print("*" * 200)
        print("Dataset Last 10 records")
        print("-----------------------------------")
        print(df.tail(10))

        print("-----------------------------------")
        print("2. Understand the Dataset")
        print("-----------------------------------")

        print("-----------------------------------")
        print("\n5. Data Types of Columns:")
        print(df.dtypes)

        print("=" * 60)

        print("6. Display column names with data types in a formatted way")
        print("\nColumn Name\t\tData Type")
        print("-" * 35)
        for column in df.columns:
            print(f"{column:<20} {df[column].dtype}")

        print("-----------------------------------")
        print("7. Dataset Summary and Information")
        print("-----------------------------------")
        df.info()
        print("\n" + "=" * 50 + "\n")

        print("8. Display Numeric Columns")
        print("-----------------------------------")
        numeric_df = df.select_dtypes(include=["int64", "float64"])
        print("Numerical Columns:")
        print(numeric_df)
        print("9. Missing Values in Numeric Attributes")
        print(numeric_df.isnull().sum())
        print("\n10. Total Missing Numeric Values:", numeric_df.isnull().sum().sum())

        float_columns = df.select_dtypes(include=["float64"]).columns
        print("-----------------------------------")
        print("10. Float Attribute Names:")
        print("-----------------------------------")
        for column in float_columns:
            print(column)
        print("11. Missing Values in Float Attributes")
        print("=" * 50)
        print(df[float_columns].isnull().sum())
        print("\n12. Total Missing Float Values:", df[float_columns].isnull().sum().sum())

        categorical_columns = [column for column in df.columns if df[column].dtype == "object"]
        categorical_df = df[categorical_columns]
        print("-----------------------------------")
        print("12. Display Categorical (Object) Attributes:")
        print("-----------------------------------")
        print(categorical_df)
        print("Missing Values in Categorical Attributes")
        print("=" * 50)
        print(categorical_df.isnull().sum())
        print("\n13. Total Missing Categorical Values:", categorical_df.isnull().sum().sum())

        print("-----------------------------------")
        print("14. Missing Values in Each Column")
        print("-" * 40)
        print(df.isnull().sum())

        total_missing = df.isnull().sum().sum()
        print("-----------------------------------")
        print("15. Total Missing Values:", total_missing)

        print("-----------------------------------")
        duplicate_count = df.duplicated().sum()
        print("16. Number of Duplicate Records:", duplicate_count)

        print("-----------------------------------")
        print("17. Statistical Overview")
        print("-----------------------------------")
        print(df.describe())

        if "CGPA" in df.columns:
            print("18. Display Histogram of CGPA Attribute")
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            plt.figure(figsize=(8.0, 5.0))
            plt.hist(df["CGPA"], bins=10, edgecolor="black")
            plt.title("Histogram of CGPA")
            plt.xlabel("CGPA")
            plt.ylabel("Frequency")
            plt.grid(True)
            plt.savefig(OUTPUT_DIR / "cgpa_histogram.png", bbox_inches="tight")
            plt.close()
            print(f"Histogram saved to: {OUTPUT_DIR / 'cgpa_histogram.png'}")
        else:
            print("18. CGPA column not found; skipping histogram.")

 except FileNotFoundError:
        print(f"Error: The file at '{DATASET_PATH}' was not found.")
 except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
 main()
