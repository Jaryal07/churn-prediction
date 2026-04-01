"""
Customer Churn Prediction with XGBoost
Dataset: Kaggle Telco Customer Churn
Tech: Python, XGBoost, LightGBM, scikit-learn, SHAP
Focus: Feature Engineering, Class Imbalance, Model Interpretability
"""

import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb

OUTPUTS_DIR = "outputs"
os.makedirs(OUTPUTS_DIR, exist_ok=True)
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    auc,
    precision_recall_curve,
    f1_score,
)
import warnings

warnings.filterwarnings("ignore")

sns.set_style("darkgrid")
plt.rcParams["figure.figsize"] = (14, 6)

MODELS_DIR = "models"
OUTPUTS_DIR = "outputs"
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)


# ==================== DATA LOADING & EXPLORATION ====================
def load_and_explore_data(file_path):
    """Load and explore customer churn dataset"""
    print("=" * 70)
    print("CUSTOMER CHURN PREDICTION - DATA EXPLORATION")
    print("=" * 70)

    df = pd.read_csv(file_path)

    print(f"\nDataset Shape: {df.shape}")
    print("\nFirst few rows:")
    print(df.head())

    print(f"\nData Types:\n{df.dtypes}")
    print(f"\nMissing Values:\n{df.isnull().sum()}")

    churn_counts = df["Churn"].value_counts()
    churn_percentage = (df["Churn"].value_counts() / len(df)) * 100

    print("\nChurn Distribution:")
    print(
        f"  No Churn: {churn_counts.get('No', 0)} ({churn_percentage.get('No', 0):.2f}%)"
    )
    print(
        f"  Churn: {churn_counts.get('Yes', 0)} ({churn_percentage.get('Yes', 0):.2f}%)"
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    churn_counts.plot(kind="bar", ax=axes[0], color=["#2ecc71", "#e74c3c"])
    axes[0].set_title("Churn Distribution", fontweight="bold")
    axes[0].set_ylabel("Count")
    axes[0].set_xticklabels(["No Churn", "Churn"], rotation=0)

    axes[1].pie(
        churn_counts,
        labels=["No Churn", "Churn"],
        autopct="%1.1f%%",
        colors=["#2ecc71", "#e74c3c"],
        startangle=90,
    )
    axes[1].set_title("Churn Percentage", fontweight="bold")

    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUTS_DIR, "churn_distribution.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    return df


# ==================== FEATURE ENGINEERING ====================
def feature_engineering(df):
    """Create and engineer features"""
    print("\n" + "=" * 70)
    print("FEATURE ENGINEERING")
    print("=" * 70)

    df_processed = df.copy()

    df_processed["TotalCharges"] = pd.to_numeric(
        df_processed["TotalCharges"], errors="coerce"
    )
    df_processed["TotalCharges"] = df_processed["TotalCharges"].fillna(
        df_processed["TotalCharges"].median()
    )

    df_processed["ChargePerMonth"] = df_processed["TotalCharges"] / (
        df_processed["tenure"] + 1
    )

    contract_mapping = {"Month-to-month": 1, "One year": 2, "Two year": 3}
    df_processed["ContractLength"] = df_processed["Contract"].map(contract_mapping)

    df_processed["NoInternetSecurity"] = (
        (df_processed["InternetService"] == "Fiber optic")
        & (df_processed["OnlineSecurity"] == "No")
    ).astype(int)

    df_processed["SeniorMonthlyCustomer"] = (
        (df_processed["SeniorCitizen"] == 1)
        & (df_processed["Contract"] == "Month-to-month")
    ).astype(int)

    df_processed["TenureGroup"] = pd.cut(
        df_processed["tenure"],
        bins=[0, 6, 12, 24, 48, 72],
        labels=["0-6 months", "6-12 months", "1-2 years", "2-4 years", "4+ years"],
        include_lowest=True,
    )

    print("✓ New features created:")
    print("  - ChargePerMonth: TotalCharges / tenure")
    print("  - ContractLength: Numeric encoding of contract type")
    print("  - NoInternetSecurity: Fiber optic without online security")
    print("  - SeniorMonthlyCustomer: Senior + month-to-month contract")
    print("  - TenureGroup: Binned tenure into categories")

    return df_processed


# ==================== DATA PREPROCESSING ====================
def preprocess_data(df):
    """Preprocess data for modeling"""
    print("\n" + "=" * 70)
    print("DATA PREPROCESSING")
    print("=" * 70)

    df_processed = df.copy()

    X = df_processed.drop(["Churn", "customerID"], axis=1)
    y = (df_processed["Churn"] == "Yes").astype(int)

    print(f"\nFeatures: {X.shape[1]}")
    print(f"Target distribution: {y.value_counts().to_dict()}")

    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

    print(f"\nCategorical features: {len(categorical_cols)}")
    print(f"Numerical features: {len(numerical_cols)}")

    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le

    numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

    if X.isnull().any().any():
        X = X.fillna(X.median(numeric_only=True))

    scaler = StandardScaler()
    X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\nTrain set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    print(f"Train churn rate: {y_train.mean():.2%}")
    print(f"Test churn rate: {y_test.mean():.2%}")

    if X_train.isnull().any().any():
        print(
            "\n⚠ NaN values detected in X_train — filling with column medians before SMOTE"
        )
        train_medians = X_train.median(numeric_only=True)
        X_train = X_train.fillna(train_medians)
        X_test = X_test.fillna(train_medians)

    smote = SMOTE(random_state=42, k_neighbors=5)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

    print("\nAfter SMOTE:")
    print(f"Train set: {X_train_smote.shape}")
    print(f"Churn ratio: {y_train_smote.mean():.2%}")

    return (
        X_train_smote,
        X_test,
        y_train_smote,
        y_test,
        X.columns,
        scaler,
        label_encoders,
    )


# ==================== XGBOOST MODEL TRAINING ====================
def train_xgboost_model(X_train, X_test, y_train, y_test, feature_names):
    """Train XGBoost model"""
    print("\n" + "=" * 70)
    print("TRAINING XGBOOST MODEL")
    print("=" * 70)

    params = {
        "objective": "binary:logistic",
        "max_depth": 7,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "gamma": 1,
        "min_child_weight": 4,
        "scale_pos_weight": 3,
        "random_state": 42,
        "n_jobs": -1,
        "eval_metric": "auc",
    }

    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=list(feature_names))
    dtest = xgb.DMatrix(X_test, label=y_test, feature_names=list(feature_names))

    evals = [(dtrain, "train"), (dtest, "eval")]
    evals_result = {}

    print("Training model with early stopping...")
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=500,
        evals=evals,
        evals_result=evals_result,
        early_stopping_rounds=50,
        verbose_eval=50,
    )

    joblib.dump(model, os.path.join(MODELS_DIR, "xgboost_model.pkl"))
    print(f"\n✓ Model trained with {model.best_iteration + 1} boosting rounds")

    return model, evals_result


# ==================== MODEL EVALUATION ====================
def evaluate_model(model, X_test, y_test, feature_names):
    """Comprehensive model evaluation"""
    print("\n" + "=" * 70)
    print("MODEL EVALUATION")
    print("=" * 70)

    dtest = xgb.DMatrix(X_test, label=y_test, feature_names=list(feature_names))
    y_pred_proba = model.predict(dtest)
    y_pred = (y_pred_proba >= 0.5).astype(int)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))

    roc_auc = roc_auc_score(y_test, y_pred_proba)
    print(f"ROC-AUC Score: {roc_auc:.4f}")

    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(f"  True Negatives: {cm[0, 0]}")
    print(f"  False Positives: {cm[0, 1]}")
    print(f"  False Negatives: {cm[1, 0]}")
    print(f"  True Positives: {cm[1, 1]}")

    f1 = f1_score(y_test, y_pred)
    print(f"\nF1 Score: {f1:.4f}")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=axes[0, 0],
        xticklabels=["No Churn", "Churn"],
        yticklabels=["No Churn", "Churn"],
    )
    axes[0, 0].set_title("Confusion Matrix", fontweight="bold")
    axes[0, 0].set_ylabel("True Label")
    axes[0, 0].set_xlabel("Predicted Label")

    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    axes[0, 1].plot(
        fpr, tpr, label=f"ROC Curve (AUC={roc_auc:.3f})", linewidth=2, color="#e74c3c"
    )
    axes[0, 1].plot([0, 1], [0, 1], "k--", label="Random Classifier")
    axes[0, 1].set_xlabel("False Positive Rate")
    axes[0, 1].set_ylabel("True Positive Rate")
    axes[0, 1].set_title("ROC Curve", fontweight="bold")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    pr_auc = auc(recall, precision)
    axes[1, 0].plot(
        recall,
        precision,
        label=f"PR Curve (AUC={pr_auc:.3f})",
        linewidth=2,
        color="#3498db",
    )
    axes[1, 0].set_xlabel("Recall")
    axes[1, 0].set_ylabel("Precision")
    axes[1, 0].set_title("Precision-Recall Curve", fontweight="bold")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].hist(
        y_pred_proba[y_test == 0],
        bins=50,
        label="No Churn (Actual)",
        alpha=0.7,
        color="#2ecc71",
    )
    axes[1, 1].hist(
        y_pred_proba[y_test == 1],
        bins=50,
        label="Churn (Actual)",
        alpha=0.7,
        color="#e74c3c",
    )
    axes[1, 1].set_xlabel("Predicted Churn Probability")
    axes[1, 1].set_ylabel("Frequency")
    axes[1, 1].set_title("Distribution of Predicted Probabilities", fontweight="bold")
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUTS_DIR, "churn_model_performance.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    return roc_auc, f1, y_pred_proba, y_pred


# ==================== FEATURE IMPORTANCE ====================
def plot_feature_importance(model, feature_names, top_n=15):
    """Plot feature importance"""
    importance_dict = model.get_score(importance_type="weight")
    importance_df = pd.DataFrame(
        {
            "Feature": list(importance_dict.keys()),
            "Importance": list(importance_dict.values()),
        }
    ).sort_values("Importance", ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    top_features = importance_df.head(top_n)
    axes[0].barh(top_features["Feature"], top_features["Importance"], color="#3498db")
    axes[0].set_xlabel("Importance (Weight)")
    axes[0].set_title(f"Top {top_n} Feature Importances", fontweight="bold")
    axes[0].invert_yaxis()

    importance_df["CumulativeImportance"] = (
        importance_df["Importance"].cumsum() / importance_df["Importance"].sum()
    )
    axes[1].plot(
        range(len(importance_df)),
        importance_df["CumulativeImportance"].values,
        marker="o",
        linewidth=2,
        color="#e74c3c",
    )
    axes[1].axhline(y=0.95, color="gray", linestyle="--", label="95% Threshold")
    axes[1].set_xlabel("Number of Features")
    axes[1].set_ylabel("Cumulative Importance")
    axes[1].set_title("Cumulative Feature Importance", fontweight="bold")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUTS_DIR, "feature_importance.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    print(f"Saved: {os.path.join(OUTPUTS_DIR, 'feature_importance.png')}")


# ==================== ANALYSIS CSV ====================
def save_analysis_csv(X_test, y_test, y_pred, y_pred_proba):
    analysis_df = X_test.copy()
    analysis_df["actual_churn"] = y_test.values
    analysis_df["predicted_churn"] = y_pred
    analysis_df["predicted_probability"] = y_pred_proba
    analysis_df.to_csv(os.path.join(OUTPUTS_DIR, "churn_analysis.csv"), index=False)
    print(f"✓ Saved analysis CSV to {os.path.join(OUTPUTS_DIR, 'churn_analysis.csv')}")


# ==================== CHURN PREDICTION ====================
def predict_churn_probability(model, customer_data, feature_names):
    """Predict churn probability for a customer"""
    dcustomer = xgb.DMatrix(customer_data, feature_names=list(feature_names))
    churn_prob = model.predict(dcustomer)[0]

    return {
        "churn_probability": churn_prob,
        "will_churn": "Yes" if churn_prob >= 0.5 else "No",
        "risk_level": (
            "High" if churn_prob >= 0.7 else "Medium" if churn_prob >= 0.5 else "Low"
        ),
    }


# ==================== MAIN EXECUTION ====================
if __name__ == "__main__":
    dataset_path = "WA_Fn-UseC_-Telco-Customer-Churn.csv"

    df = load_and_explore_data(dataset_path)
    df_engineered = feature_engineering(df)
    X_train, X_test, y_train, y_test, feature_names, scaler, label_encoders = (
        preprocess_data(df_engineered)
    )
    model, evals_result = train_xgboost_model(
        X_train, X_test, y_train, y_test, feature_names
    )
    roc_auc, f1, y_pred_proba, y_pred = evaluate_model(
        model, X_test, y_test, feature_names
    )
    importance_df = plot_feature_importance(model, feature_names, top_n=15)
    save_analysis_csv(X_test, y_test, y_pred, y_pred_proba)

    print("\n" + "=" * 70)
    print("✓ Customer Churn Prediction Model Complete!")
    print(f"✓ Model saved to {os.path.join(MODELS_DIR, 'xgboost_model.pkl')}")
    print(f"✓ Scaler saved to {os.path.join(MODELS_DIR, 'scaler.pkl')}")
    print(f"✓ Outputs saved in {OUTPUTS_DIR}/")
    print("=" * 70)
