import os
import tempfile
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StringType
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression, DecisionTreeClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler


# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="PySpark Classification Analysis and Prediction Platform",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("PySpark Classification Analysis and Prediction Platform")
st.markdown(
    """

"""
)


# =========================================================
# SPARK SESSION
# =========================================================
@st.cache_resource

def get_spark():
    spark = (
        SparkSession.builder
        .appName("Classification_Pyspark_app")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.executor.memory", "2g")
        .config("spark.driver.memory", "2g")
        .config("spark.python.worker.memory", "512m")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    return spark


spark = get_spark()


# =========================================================
# HELPERS
# =========================================================
def safe_read_file(uploaded_file, ext: str):
    suffix = f".{ext}" if not ext.startswith(".") else ext
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getvalue())
        return tmp.name


@st.cache_data(show_spinner=False)
def excel_sheet_names(file_bytes: bytes) -> List[str]:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp:
        tmp.write(file_bytes)
        temp_path = tmp.name
    try:
        xls = pd.ExcelFile(temp_path)
        return xls.sheet_names
    finally:
        try:
            os.unlink(temp_path)
        except Exception:
            pass


def read_data_with_spark(temp_path: str, file_type: str, excel_sheet: Optional[str] = None):
    if file_type == "csv":
        return spark.read.csv(temp_path, header=True, inferSchema=True)

    if file_type == "excel":
        # Read Excel using pandas first, then convert to Spark.
        pdf = pd.read_excel(temp_path, sheet_name=excel_sheet)
        pdf.columns = [str(c).strip() for c in pdf.columns]
        return spark.createDataFrame(pdf)

    if file_type == "libsvm":
        return spark.read.format("libsvm").load(temp_path)

    raise ValueError("Unsupported file type")


def identify_column_types(df, target_col: str) -> Tuple[List[str], List[str]]:
    numeric_cols = []
    categorical_cols = []

    for field in df.schema.fields:
        col_name = field.name
        if col_name == target_col:
            continue

        dtype_str = str(field.dataType).lower()
        if any(x in dtype_str for x in ["int", "double", "float", "long", "short", "decimal"]):
            numeric_cols.append(col_name)
        else:
            categorical_cols.append(col_name)

    return numeric_cols, categorical_cols


def cast_target_to_numeric(df, target_col: str):
    dtype_str = str(df.schema[target_col].dataType).lower()

    if any(x in dtype_str for x in ["int", "double", "float", "long", "short", "decimal"]):
        return df.withColumn(target_col, F.col(target_col).cast("double"))

    label_indexer = StringIndexer(inputCol=target_col, outputCol="label", handleInvalid="keep")
    model = label_indexer.fit(df)
    transformed = model.transform(df)
    labels = list(model.labels)
    return transformed.drop(target_col).withColumnRenamed("label", target_col), labels


def prepare_dataframe(df, target_col: str, selected_features: List[str]):
    df = df.select([target_col] + selected_features)

    # Standardize column names for safety
    for c in df.columns:
        safe_c = c.strip().replace(" ", "_").replace("/", "_").replace("-", "_")
        if safe_c != c:
            df = df.withColumnRenamed(c, safe_c)

    target_col = target_col.strip().replace(" ", "_").replace("/", "_").replace("-", "_")
    selected_features = [c.strip().replace(" ", "_").replace("/", "_").replace("-", "_") for c in selected_features]

    # Drop rows with missing target
    df = df.dropna(subset=[target_col])

    numeric_cols, categorical_cols = identify_column_types(df, target_col)
    numeric_cols = [c for c in selected_features if c in numeric_cols]
    categorical_cols = [c for c in selected_features if c in categorical_cols]

    # Cast numeric columns
    for c in numeric_cols:
        df = df.withColumn(c, F.col(c).cast("double"))

    df = cast_missing_values(df, numeric_cols, categorical_cols)
    return df, target_col, numeric_cols, categorical_cols, selected_features



def cast_missing_values(df, numeric_cols: List[str], categorical_cols: List[str]):
    for c in numeric_cols:
        median_approx = df.approxQuantile(c, [0.5], 0.01)
        fill_val = median_approx[0] if median_approx and len(median_approx) > 0 else 0.0
        df = df.withColumn(c, F.when(F.col(c).isNull(), F.lit(float(fill_val))).otherwise(F.col(c)))

    for c in categorical_cols:
        df = df.withColumn(c, F.when(F.col(c).isNull(), F.lit("Unknown")).otherwise(F.col(c).cast(StringType())))

    return df



def build_feature_pipeline(target_col: str, numeric_cols: List[str], categorical_cols: List[str], classifier):
    stages = []
    encoded_cols = []

    for c in categorical_cols:
        idx_col = f"{c}_idx"
        vec_col = f"{c}_vec"
        indexer = StringIndexer(inputCol=c, outputCol=idx_col, handleInvalid="keep")
        encoder = OneHotEncoder(inputCol=idx_col, outputCol=vec_col)
        stages.extend([indexer, encoder])
        encoded_cols.append(vec_col)

    assembler_inputs = numeric_cols + encoded_cols
    assembler = VectorAssembler(inputCols=assembler_inputs, outputCol="features", handleInvalid="keep")
    stages.append(assembler)
    stages.append(classifier)

    return Pipeline(stages=stages)



def evaluate_pyspark_model(predictions, label_col: str = "label"):
    metric_results = {}

    try:
        acc_eval = MulticlassClassificationEvaluator(
            labelCol=label_col,
            predictionCol="prediction",
            metricName="accuracy",
        )
        metric_results["accuracy"] = acc_eval.evaluate(predictions)
    except Exception:
        metric_results["accuracy"] = None

    try:
        f1_eval = MulticlassClassificationEvaluator(
            labelCol=label_col,
            predictionCol="prediction",
            metricName="f1",
        )
        metric_results["f1"] = f1_eval.evaluate(predictions)
    except Exception:
        metric_results["f1"] = None

    try:
        auc_eval = BinaryClassificationEvaluator(
            labelCol=label_col,
            rawPredictionCol="rawPrediction",
            metricName="areaUnderROC",
        )
        metric_results["auc"] = auc_eval.evaluate(predictions)
    except Exception:
        metric_results["auc"] = None

    return metric_results



def spark_prediction_table(predictions, label_col: str):
    cols = [label_col, "prediction"]
    if "probability" in predictions.columns:
        cols.append("probability")

    pdf = predictions.select(*cols).coalesce(1).toPandas()
    pdf = pdf.rename(columns={label_col: "Actual", "prediction": "Predicted"})

    if "probability" in pdf.columns:
        try:
            pdf["Probability"] = pdf["probability"].apply(lambda x: [float(v) for v in x])
            pdf.drop(columns=["probability"], inplace=True)
        except Exception:
            pass

    return pdf



def fit_knn_with_preprocessed_data(df, target_col, numeric_cols, categorical_cols, n_neighbors, train_ratio, random_seed):
    train_df, test_df = df.randomSplit([train_ratio, 1 - train_ratio], seed=int(random_seed))

    indexers = []
    encoders = []
    encoded_cols = []

    for c in categorical_cols:
        idx_col = f"{c}_idx"
        vec_col = f"{c}_vec"
        indexers.append(StringIndexer(inputCol=c, outputCol=idx_col, handleInvalid="keep"))
        encoders.append(OneHotEncoder(inputCol=idx_col, outputCol=vec_col))
        encoded_cols.append(vec_col)

    assembler_inputs = numeric_cols + encoded_cols
    assembler = VectorAssembler(inputCols=assembler_inputs, outputCol="features", handleInvalid="keep")

    prep_pipeline = Pipeline(stages=indexers + encoders + [assembler])
    prep_model = prep_pipeline.fit(train_df)

    train_prepared = prep_model.transform(train_df).select(target_col, "features")
    test_prepared = prep_model.transform(test_df).select(target_col, "features")

    train_pdf = train_prepared.coalesce(1).toPandas()
    test_pdf = test_prepared.coalesce(1).toPandas()

    X_train = np.vstack(train_pdf["features"].apply(lambda x: np.array(x)).values)
    y_train = train_pdf[target_col].astype(int).values

    X_test = np.vstack(test_pdf["features"].apply(lambda x: np.array(x)).values)
    y_test = test_pdf[target_col].astype(int).values

    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    y_prob = knn.predict_proba(X_test) if hasattr(knn, "predict_proba") else None

    result_pdf = pd.DataFrame({
        "Actual": y_test,
        "Predicted": y_pred,
    })

    if y_prob is not None and y_prob.shape[1] >= 2:
        result_pdf["Probability_Class_1"] = y_prob[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1": None,
        "auc": None,
        "rows_tested": int(len(result_pdf)),
    }

    return knn, prep_model, result_pdf, metrics



def manual_input_ui(selected_features, numeric_cols, categorical_cols, df):
    st.subheader("Manual Prediction Input")
    values = {}

    columns_layout = st.columns(3)
    idx = 0
    for col in selected_features:
        with columns_layout[idx % 3]:
            if col in numeric_cols:
                values[col] = st.number_input(col, value=0.0, format="%.6f")
            else:
                options = [str(r[col]) for r in df.select(col).distinct().limit(200).collect()]
                options = sorted(list(set(options))) if options else ["Unknown"]
                values[col] = st.selectbox(col, options)
        idx += 1

    return values



def build_single_input_dataframe(input_values: dict):
    return pd.DataFrame([input_values])


# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.header("Upload & Settings")

uploaded_file = st.sidebar.file_uploader(
    "Upload your dataset",
    type=["csv", "xlsx", "xls", "txt", "data", "libsvm"],
)

train_ratio = st.sidebar.slider("Train Ratio", 0.5, 0.9, 0.7, 0.05)
random_seed = st.sidebar.number_input("Random Seed", min_value=1, value=42, step=1)
model_choice = st.sidebar.selectbox(
    "Select Model",
    ["Logistic Regression", "Decision Tree", "KNN"],
)

# Model-specific parameters
st.sidebar.subheader("Model Parameters")
reg_param = st.sidebar.number_input("Logistic regParam", min_value=0.0, value=0.0, step=0.01, format="%.4f")
elastic_net = st.sidebar.slider("Logistic elasticNetParam", 0.0, 1.0, 0.0, 0.05)
max_iter = st.sidebar.slider("Logistic maxIter", 10, 300, 100, 10)
max_depth = st.sidebar.slider("Decision Tree maxDepth", 2, 20, 5, 1)
knn_k = st.sidebar.slider("KNN k", 1, 25, 5, 1)


# =========================================================
# MAIN LOGIC
# =========================================================
if uploaded_file is None:
    st.info("Upload a CSV, Excel, or LIBSVM file to start.")
    st.stop()

file_name = uploaded_file.name.lower()
ext = file_name.split(".")[-1]

file_type = None
excel_sheet = None
temp_path = None

try:
    if ext == "csv":
        file_type = "csv"
        temp_path = safe_read_file(uploaded_file, ".csv")

    elif ext in ["xlsx", "xls"]:
        file_type = "excel"
        sheet_names = excel_sheet_names(uploaded_file.getvalue())
        excel_sheet = st.sidebar.selectbox("Excel Sheet", sheet_names)
        temp_path = safe_read_file(uploaded_file, ".xlsx")

    elif ext in ["txt", "data", "libsvm"]:
        file_type = "libsvm"
        temp_path = safe_read_file(uploaded_file, ".txt")

    else:
        st.error("Unsupported file type.")
        st.stop()

    df = read_data_with_spark(temp_path, file_type, excel_sheet)

    st.subheader("Dataset Preview")
    st.dataframe(df.limit(30).coalesce(1).toPandas(), use_container_width=True)

    st.subheader("Dataset Overview")
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", df.count())
    c2.metric("Columns", len(df.columns))
    c3.metric("File Type", file_type.upper())

    if file_type == "libsvm":
        st.info(
            "Detected LIBSVM format. The dataset already contains `label` and `features`. "
            "For this format, the app will directly use the existing vector column."
        )

        target_col = "label"
        train_df, test_df = df.randomSplit([train_ratio, 1 - train_ratio], seed=int(random_seed))

        if model_choice == "Logistic Regression":
            clf = LogisticRegression(
                featuresCol="features",
                labelCol="label",
                predictionCol="prediction",
                probabilityCol="probability",
                rawPredictionCol="rawPrediction",
                regParam=float(reg_param),
                elasticNetParam=float(elastic_net),
                maxIter=int(max_iter),
            )
            model = clf.fit(train_df)
            predictions = model.transform(test_df)
            metrics = evaluate_pyspark_model(predictions, label_col="label")
            pred_pdf = spark_prediction_table(predictions, label_col="label")

        elif model_choice == "Decision Tree":
            clf = DecisionTreeClassifier(
                featuresCol="features",
                labelCol="label",
                predictionCol="prediction",
                probabilityCol="probability",
                rawPredictionCol="rawPrediction",
                maxDepth=int(max_depth),
            )
            model = clf.fit(train_df)
            predictions = model.transform(test_df)
            metrics = evaluate_pyspark_model(predictions, label_col="label")
            pred_pdf = spark_prediction_table(predictions, label_col="label")

        else:
            train_pdf = train_df.select("label", "features").coalesce(1).toPandas()
            test_pdf = test_df.select("label", "features").coalesce(1).toPandas()

            X_train = np.vstack(train_pdf["features"].apply(lambda x: np.array(x)).values)
            y_train = train_pdf["label"].astype(int).values
            X_test = np.vstack(test_pdf["features"].apply(lambda x: np.array(x)).values)
            y_test = test_pdf["label"].astype(int).values

            model = KNeighborsClassifier(n_neighbors=knn_k)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None

            pred_pdf = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
            if y_prob is not None and y_prob.shape[1] >= 2:
                pred_pdf["Probability_Class_1"] = y_prob[:, 1]

            metrics = {
                "accuracy": float(accuracy_score(y_test, y_pred)),
                "f1": None,
                "auc": None,
            }

        st.subheader("Model Metrics")
        m1, m2, m3 = st.columns(3)
        m1.metric("Accuracy", f"{metrics['accuracy']:.4f}" if metrics["accuracy"] is not None else "N/A")
        m2.metric("F1 Score", f"{metrics['f1']:.4f}" if metrics["f1"] is not None else "N/A")
        m3.metric("AUC", f"{metrics['auc']:.4f}" if metrics["auc"] is not None else "N/A")

        st.subheader("Prediction Results")
        st.dataframe(pred_pdf.head(200), use_container_width=True)
        st.bar_chart(pred_pdf["Predicted"].value_counts().sort_index())

        csv_bytes = pred_pdf.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Predictions CSV",
            data=csv_bytes,
            file_name="predictions_output.csv",
            mime="text/csv",
        )

        st.warning(
            "Manual single-row prediction for LIBSVM is not included here because the input vector length depends on the source file structure. "
            "For manual scoring, CSV or Excel format is more convenient."
        )

    else:
        st.subheader("Column Selection")
        target_col = st.selectbox("Select target column", df.columns)
        feature_candidates = [c for c in df.columns if c != target_col]
        selected_features = st.multiselect(
            "Select feature columns",
            feature_candidates,
            default=feature_candidates[: min(8, len(feature_candidates))],
        )

        if not selected_features:
            st.warning("Please select at least one feature column.")
            st.stop()

        original_target = target_col
        prepared_df, target_col, numeric_cols, categorical_cols, selected_features = prepare_dataframe(df, target_col, selected_features)

        # Convert target if categorical
        target_labels = None
        dtype_str = str(prepared_df.schema[target_col].dataType).lower()
        if not any(x in dtype_str for x in ["int", "double", "float", "long", "short", "decimal"]):
            label_indexer = StringIndexer(inputCol=target_col, outputCol="label", handleInvalid="keep")
            label_model = label_indexer.fit(prepared_df)
            target_labels = list(label_model.labels)
            prepared_df = label_model.transform(prepared_df).drop(target_col).withColumnRenamed("label", target_col)

        else:
            prepared_df = prepared_df.withColumn(target_col, F.col(target_col).cast("double"))

        st.write({
            "Target": original_target,
            "Processed Target": target_col,
            "Numeric Features": numeric_cols,
            "Categorical Features": categorical_cols,
        })

        if model_choice == "Logistic Regression":
            classifier = LogisticRegression(
                featuresCol="features",
                labelCol=target_col,
                predictionCol="prediction",
                probabilityCol="probability",
                rawPredictionCol="rawPrediction",
                regParam=float(reg_param),
                elasticNetParam=float(elastic_net),
                maxIter=int(max_iter),
            )
            pipeline = build_feature_pipeline(target_col, numeric_cols, categorical_cols, classifier)
            train_df, test_df = prepared_df.randomSplit([train_ratio, 1 - train_ratio], seed=int(random_seed))
            model = pipeline.fit(train_df)
            predictions = model.transform(test_df)
            metrics = evaluate_pyspark_model(predictions, label_col=target_col)
            pred_pdf = spark_prediction_table(predictions, label_col=target_col)

        elif model_choice == "Decision Tree":
            classifier = DecisionTreeClassifier(
                featuresCol="features",
                labelCol=target_col,
                predictionCol="prediction",
                probabilityCol="probability",
                rawPredictionCol="rawPrediction",
                maxDepth=int(max_depth),
            )
            pipeline = build_feature_pipeline(target_col, numeric_cols, categorical_cols, classifier)
            train_df, test_df = prepared_df.randomSplit([train_ratio, 1 - train_ratio], seed=int(random_seed))
            model = pipeline.fit(train_df)
            predictions = model.transform(test_df)
            metrics = evaluate_pyspark_model(predictions, label_col=target_col)
            pred_pdf = spark_prediction_table(predictions, label_col=target_col)

        else:
            model, prep_model, pred_pdf, metrics = fit_knn_with_preprocessed_data(
                prepared_df,
                target_col,
                numeric_cols,
                categorical_cols,
                n_neighbors=int(knn_k),
                train_ratio=float(train_ratio),
                random_seed=int(random_seed),
            )

        st.subheader("Model Metrics")
        m1, m2, m3 = st.columns(3)
        m1.metric("Accuracy", f"{metrics['accuracy']:.4f}" if metrics.get("accuracy") is not None else "N/A")
        m2.metric("F1 Score", f"{metrics['f1']:.4f}" if metrics.get("f1") is not None else "N/A")
        m3.metric("AUC", f"{metrics['auc']:.4f}" if metrics.get("auc") is not None else "N/A")

        st.subheader("Prediction Results")
        st.dataframe(pred_pdf.head(200), use_container_width=True)
        st.bar_chart(pred_pdf["Predicted"].value_counts().sort_index())

        st.subheader("Confusion Matrix")
        try:
            cm = confusion_matrix(pred_pdf["Actual"], pred_pdf["Predicted"])
            cm_df = pd.DataFrame(cm)
            st.dataframe(cm_df, use_container_width=True)
        except Exception:
            st.info("Confusion matrix could not be generated.")

        csv_bytes = pred_pdf.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Predictions CSV",
            data=csv_bytes,
            file_name="predictions_output.csv",
            mime="text/csv",
        )

        # Manual prediction section
        manual_values = manual_input_ui(selected_features, numeric_cols, categorical_cols, prepared_df)

        if st.button("Predict Manual Input"):
            manual_pdf = build_single_input_dataframe(manual_values)
            manual_spark = spark.createDataFrame(manual_pdf)

            for c in numeric_cols:
                if c in manual_spark.columns:
                    manual_spark = manual_spark.withColumn(c, F.col(c).cast("double"))
            for c in categorical_cols:
                if c in manual_spark.columns:
                    manual_spark = manual_spark.withColumn(c, F.col(c).cast("string"))

            if model_choice in ["Logistic Regression", "Decision Tree"]:
                manual_pred = model.transform(manual_spark)
                out = manual_pred.select("prediction", "probability" if "probability" in manual_pred.columns else "prediction").coalesce(1).toPandas()
                st.success(f"Predicted class: {out.iloc[0]['prediction']}")
                if "probability" in out.columns:
                    st.write({"Probability": [float(v) for v in out.iloc[0]["probability"]]})
            else:
                transformed_manual = prep_model.transform(manual_spark).select("features").coalesce(1).toPandas()
                X_new = np.vstack(transformed_manual["features"].apply(lambda x: np.array(x)).values)
                pred = model.predict(X_new)[0]
                st.success(f"Predicted class: {pred}")
                if hasattr(model, "predict_proba"):
                    prob = model.predict_proba(X_new)[0]
                    st.write({"Probability": [float(v) for v in prob]})

finally:
    if temp_path is not None:
        try:
            os.unlink(temp_path)
        except Exception:
            pass
