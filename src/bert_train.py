import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments
)

from datasets import Dataset

# Load dataset
df = pd.read_csv("data/processed/processed_reviews.csv")

# Remove missing values
df = df.dropna(subset=["Review Text", "Sentiment"])

# Features and labels
X = df["Review Text"]
y = df["Sentiment"]

# Convert labels to numbers
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

print("Label Mapping:")
for i, label in enumerate(label_encoder.classes_):
    print(f"{label} -> {i}")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Create Hugging Face datasets
train_dataset = Dataset.from_dict({"text": X_train.tolist(), "label": y_train.tolist()})
test_dataset = Dataset.from_dict({"text": X_test.tolist(), "label": y_test.tolist()})

# Load tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Tokenization function
def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=64)

train_dataset = train_dataset.map(tokenize, batched=True)
test_dataset = test_dataset.map(tokenize, batched=True)

train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# Load BERT model
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)

# Metrics function
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)

    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds ,average="weighted")

    acc = accuracy_score(labels, preds)

    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

# Training settings
training_args = TrainingArguments(
    output_dir="./bert_results",
    eval_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=1,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    logging_dir="./logs",
    logging_steps=50
)

#Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

#Train the model
trainer.train()

#Evaluate the model
eval_results = trainer.evaluate()
print(f"Evaluation results: {eval_results}")

# Save model
model.save_pretrained("saved_model")

# Save tokenizer
tokenizer.save_pretrained("saved_model")

print("Model saved successfully!")