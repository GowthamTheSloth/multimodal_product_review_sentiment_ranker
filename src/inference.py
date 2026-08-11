import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Path to trained model checkpoint
MODEL_PATH = "bert_results/checkpoint-2106"

# Load tokenizer and model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained(MODEL_PATH)

# Put model in evaluation mode
model.eval()

# Label mapping
id_to_label = {
    0: "negative",
    1: "neutral",
    2: "positive"
}

while True:
    review = input("\nEnter review (or type 'quit'): ")

    if review.lower() == "quit":
        break

    inputs = tokenizer(
        review,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=64
    )

    print("Input IDs:", inputs["input_ids"][0][:15])

    with torch.no_grad():
        outputs = model(**inputs)

    print("Logits:", outputs.logits)

    prediction = torch.argmax(outputs.logits, dim=1).item()

    print("Raw prediction:", prediction)
    print("Predicted Sentiment:", id_to_label[prediction])
