from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained(
    "bert_results/checkpoint-2106"
)

print(tokenizer.special_tokens_map)
print("Vocab size:", tokenizer.vocab_size)