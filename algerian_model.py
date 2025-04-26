# model_handler.py
import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn as nn
import pandas as pd

try:
    # Load topic mapping
    df = pd.read_excel("Arabic.xlsx")
    topics = df["Topic"].unique()
    topic_to_id = {topic: i for i, topic in enumerate(topics)}
    id_to_topic = {i: topic for topic, i in topic_to_id.items()}
except Exception as e:
    print(f"Error loading topic mapping: {e}")
    topics = []
    topic_to_id = {}
    id_to_topic = {}

try:
    # Load model
    model_name = "aubmindlab/bert-base-arabertv02"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    class MultiTaskModel(nn.Module):
        def __init__(self, model_name, num_topics):
            super(MultiTaskModel, self).__init__()
            self.bert = AutoModel.from_pretrained(model_name)
            self.hate_speech_classifier = nn.Linear(self.bert.config.hidden_size, 2)
            self.topic_classifier = nn.Linear(self.bert.config.hidden_size, num_topics)

        def forward(self, input_ids, attention_mask):
            bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            pooled_output = bert_outputs.last_hidden_state[:, 0, :]
            hate_speech_logits = self.hate_speech_classifier(pooled_output)
            topic_logits = self.topic_classifier(pooled_output)
            return hate_speech_logits, topic_logits

    num_topics = len(topics)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiTaskModel(model_name, num_topics).to(device)
    model.load_state_dict(torch.load("models/arabert_hate_speech_topics (1).pth", map_location=device))
    model.eval()
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

def predict(text: str):
    if not model:
        raise ValueError("Model not loaded")

    inputs = tokenizer(
        text,
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        hate_speech_logits, topic_logits = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"]
        )

    hate_pred = torch.argmax(hate_speech_logits, dim=1).item()
    topic_pred = torch.argmax(topic_logits, dim=1).item()

    return {
        "hate_speech": "Hate Speech" if hate_pred == 1 else "Not Hate Speech",
        "topic": id_to_topic.get(topic_pred, "Unknown")
    }
