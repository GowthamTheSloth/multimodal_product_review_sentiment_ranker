import torch
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, f1_score, classification_report

mlflow.set_experiment("sentiment_ablation")


FUSED_PATH = "data/processed/fused_embeddings.pt"
CSV_PATH = "data/processed/multimodal_reviews.csv"

print("Loading fused embeddings...")

fused_data = torch.load(FUSED_PATH, weights_only=False)

fused_embeddings = fused_data["fused_embeddings"]
original_indices = fused_data["original_indices"]

print(f"Fused embeddings shape: {fused_embeddings.shape}")

# Load the multimodal dataset
df = pd.read_csv(CSV_PATH)

# Select sentiment labels corresponding to the original CSV indices
ratings = df.iloc[original_indices]["rating"].values

labels = []

for rating in ratings:
    if rating <= 2:
        labels.append("negative")
    elif rating == 3:
        labels.append("neutral")
    else:
        labels.append("positive")

print(f"Labels shape: {len(labels)}")

# Convert PyTorch tensor to NumPy
X = fused_embeddings.numpy()
y = labels

print(f"X shape: {X.shape}")
print(f"Number of labels: {len(y)}")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"\nTraining samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

# Train classifier
print("\nTraining multimodal classifier...")

with mlflow.start_run(run_name="bert_clip_fusion"):

    mlflow.log_param("feature_set", "BERT+CLIP fusion")
    mlflow.log_param("embedding_dim", X.shape[1])
    mlflow.log_param("classifier", "LogisticRegression")
    mlflow.log_param("max_iter", 2000)
    mlflow.log_param("class_weight", "balanced")
    mlflow.log_param("test_size", 0.2)
    mlflow.log_param("random_state", 42)

    classifier = LogisticRegression(max_iter=2000, class_weight="balanced")

    classifier.fit(X_train, y_train)

    # Predictions
    y_pred = classifier.predict(X_test)

    # Evaluation
    accuracy = accuracy_score(y_test, y_pred)

    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="weighted")
    f1_macro = f1_score(y_test, y_pred, average="macro")

    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision_weighted", precision)
    mlflow.log_metric("recall_weighted", recall)
    mlflow.log_metric("f1_weighted", f1)
    mlflow.log_metric("f1_macro", f1_macro)

    report_text = classification_report(y_test, y_pred)
    mlflow.log_text(report_text, "classification_report.txt")

    mlflow.sklearn.log_model(classifier, "model")

    print("\n===== Multimodal Model Results =====")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"F1 Macro:  {f1_macro:.4f}")

    print("\nClassification Report:")
    print(report_text)