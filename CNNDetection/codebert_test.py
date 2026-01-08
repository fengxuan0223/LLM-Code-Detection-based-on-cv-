import torch
from transformers import RobertaTokenizer, RobertaModel

MODEL_NAME = r"D:/models/codebert-base"

def main():
    tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)
    model = RobertaModel.from_pretrained(MODEL_NAME)

    code = """
    def add(a, b):
        return a + b
    """

    inputs = tokenizer(
        code,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
    )

    with torch.no_grad():
        outputs = model(**inputs)

    cls_embedding = outputs.last_hidden_state[:, 0, :]  # [1, 768]
    print("CLS shape:", cls_embedding.shape)

if __name__ == "__main__":
    main()
