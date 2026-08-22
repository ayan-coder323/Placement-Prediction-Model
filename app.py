from pathlib import Path

import pandas as pd
from flask import Flask, abort, render_template, send_from_directory, url_for


BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
OUTPUT_DIR = BASE_DIR / "Outputs"
EDA_DIR = OUTPUT_DIR / "EDA_Analysis"
BOXPLOT_DIR = OUTPUT_DIR / "Boxplots_correlation"

RAW_DATASET = DATASET_DIR / "placement_predict_50K_Raw.csv"

CLEAN_DATASETS = [
    {
        "title": "Deletion, Mean, Median, and Model Imputation",
        "file": DATASET_DIR / "clean_del_mean_model_M2.csv",
        "description": "Combines deletion, mean, median, model-based imputation, and missing indicators.",
    },
    {
        "title": "Label Encoding",
        "file": DATASET_DIR / "clean_label_encode_M2.csv",
        "description": "Cleans and label-encodes categorical columns.",
    },
    {
        "title": "Embedding Encoding",
        "file": DATASET_DIR / "clean_embedded_encode_M2.csv",
        "description": "Creates simple embedding-style vectors for categorical values.",
    },
    {
        "title": "Ordinal Encoding",
        "file": DATASET_DIR / "clean_ordinal_encode_M2.csv",
        "description": "Applies ordinal encoding after cleaning missing values.",
    },
    {
        "title": "One-Hot Encoding",
        "file": DATASET_DIR / "clean_one_hot_encoding_M2.csv",
        "description": "Creates one-hot encoded features from categorical columns.",
    },
    {
        "title": "Target Encoding",
        "file": DATASET_DIR / "clean_target_encode_M2.csv",
        "description": "Replaces categories with target-based mean values.",
    },
    {
        "title": "Standardization, Min-Max Scaling, and Normalization",
        "file": DATASET_DIR / "clean_minmax_stand_norma_M2.csv",
        "description": "Adds standardized, scaled, and normalized variants of numeric columns.",
    },
]

VISUALIZATION_SECTIONS = [
    {
        "title": "EDA Overview",
        "files": [
            EDA_DIR / "Correlation_Heatmap.png",
            EDA_DIR / "Missing_Values_Heatmap.png",
            EDA_DIR / "Target_Distribution.png",
            EDA_DIR / "scatterplot.png",
            EDA_DIR / "Pairplot.png",
        ],
    },
    {
        "title": "Key Boxplots",
        "files": [
            EDA_DIR / "CGPA_boxplot.png",
            EDA_DIR / "AttendancePercent_boxplot.png",
            EDA_DIR / "AptitudeTestScore_boxplot.png",
            EDA_DIR / "CodingTestScore_boxplot.png",
        ],
    },
    {
        "title": "Placement vs Feature Boxplots",
        "files": [
            BOXPLOT_DIR / "boxplot_CGPA_vs_PlacementStatus.png",
            BOXPLOT_DIR / "boxplot_AttendancePercent_vs_PlacementStatus.png",
            BOXPLOT_DIR / "boxplot_AptitudeTestScore_vs_PlacementStatus.png",
            BOXPLOT_DIR / "boxplot_CodingTestScore_vs_PlacementStatus.png",
        ],
    },
]

app = Flask(__name__, template_folder="template", static_folder="static")


def read_csv_info(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset file: {path}")

    df = pd.read_csv(path)
    numeric_columns = df.select_dtypes(include=["number"]).columns.tolist()
    categorical_columns = df.select_dtypes(include=["object", "str"]).columns.tolist()

    missing_by_column = (
        df.isnull().sum().sort_values(ascending=False).reset_index().rename(
            columns={"index": "column", 0: "missing"}
        )
    ).to_dict(orient="records")

    return {
        "name": path.name,
        "path": str(path),
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "numeric_count": len(numeric_columns),
        "categorical_count": len(categorical_columns),
        "missing_total": int(df.isnull().sum().sum()),
        "duplicate_total": int(df.duplicated().sum()),
        "head": df.head(8).to_dict(orient="records"),
        "columns_list": df.columns.tolist(),
        "missing_by_column": missing_by_column,
    }


def load_frame(path: Path, rows: int = 10) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset file: {path}")
    return pd.read_csv(path).head(rows)


def build_dashboard_stats() -> list[dict]:
    raw = read_csv_info(RAW_DATASET)
    return [
        {"label": "Total Students", "value": raw["rows"]},
        {"label": "Total Features", "value": raw["columns"]},
        {"label": "Numeric Features", "value": raw["numeric_count"]},
        {"label": "Categorical Features", "value": raw["categorical_count"]},
        {"label": "Missing Values", "value": raw["missing_total"]},
        {"label": "Duplicate Records", "value": raw["duplicate_total"]},
    ]


def asset_url(path: Path) -> str:
    return url_for("artifact", filename=str(path.relative_to(OUTPUT_DIR)).replace("\\", "/"))


def image_card(path: Path, caption: str) -> dict:
    return {"caption": caption, "url": asset_url(path)}


@app.route("/artifacts/<path:filename>")
def artifact(filename: str):
    full_path = (OUTPUT_DIR / filename).resolve()
    if OUTPUT_DIR not in full_path.parents and full_path != OUTPUT_DIR:
        abort(404)
    if not full_path.exists():
        abort(404)
    return send_from_directory(OUTPUT_DIR, filename)


@app.route("/")
def home():
    raw = read_csv_info(RAW_DATASET)
    return render_template(
        "home.html",
        dashboard_stats=build_dashboard_stats(),
        raw_dataset=raw,
        clean_datasets=CLEAN_DATASETS,
    )


@app.route("/about")
def about():
    return render_template("about.html", raw_dataset=read_csv_info(RAW_DATASET))


@app.route("/dataset")
def dataset():
    raw = read_csv_info(RAW_DATASET)
    return render_template(
        "dataset.html",
        raw_dataset=raw,
        sample_rows=load_frame(RAW_DATASET, 10).to_dict(orient="records"),
        clean_datasets=[
            {**item, **read_csv_info(item["file"])}
            for item in CLEAN_DATASETS
        ],
    )


@app.route("/preprocessing")
def preprocessing():
    return render_template(
        "preprocessing.html",
        preprocessing_outputs=[
            {**item, **read_csv_info(item["file"])}
            for item in CLEAN_DATASETS
        ],
    )


@app.route("/visualization")
def visualization():
    sections = []
    for section in VISUALIZATION_SECTIONS:
        images = []
        for file_path in section["files"]:
            if file_path.exists():
                images.append(
                    image_card(
                        file_path,
                        file_path.stem.replace("_", " ").replace("boxplot ", "").strip(),
                    )
                )
        sections.append({"title": section["title"], "images": images})

    return render_template("visualization.html", visualization_sections=sections)


@app.route("/models")
def models():
    return render_template(
        "models.html",
        model_note="Model training scripts are not present yet, so this page remains a placeholder.",
    )


@app.route("/prediction")
def prediction():
    return render_template("prediction.html")


@app.route("/dashboard")
def dashboard():
    raw = read_csv_info(RAW_DATASET)
    return render_template(
        "dashboard.html",
        dashboard_stats=build_dashboard_stats(),
        raw_dataset=raw,
    )


@app.route("/reports")
def reports():
    summary_file = OUTPUT_DIR / "EDA_Analysis" / "Statistical_Summary.csv"
    correlation_file = OUTPUT_DIR / "EDA_Analysis" / "Correlation_Matrix.csv"
    summary_rows = []
    summary_headers = []
    correlation_highlights = []

    if summary_file.exists():
        summary_df = pd.read_csv(summary_file).head(12)
        summary_rows = summary_df.to_dict(orient="records")
        summary_headers = summary_df.columns.tolist()

    if correlation_file.exists():
        corr = pd.read_csv(correlation_file, index_col=0)
        if "PlacementStatus" in corr.columns:
            ranking = (
                corr["PlacementStatus"]
                .drop(labels=["PlacementStatus"], errors="ignore")
                .abs()
                .sort_values(ascending=False)
                .head(10)
            )
            correlation_highlights = [
                {"feature": index, "score": float(value)}
                for index, value in ranking.items()
            ]

    return render_template(
        "reports.html",
        summary_rows=summary_rows,
        summary_headers=summary_headers,
        correlation_highlights=correlation_highlights,
        report_files=[
            summary_file,
            correlation_file,
        ],
    )


@app.route("/contact")
def contact():
    return render_template("contact.html")


if __name__ == "__main__":
    app.run(debug=True)
