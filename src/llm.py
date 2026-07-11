
from google.colab import drive
import sys
import torch

drive.mount('/content/drive')
sys.path.append(
"/content/drive/MyDrive/SupportAI"
)

import sys

sys.path.insert(0, "/content/drive/MyDrive/SupportAI/src")

from inference import predict_all
from transformers import AutoModelForSequenceClassification, AutoTokenizer

path = "/content/drive/MyDrive/SupportAI/models"

def load_models(path):

    tokenizer = AutoTokenizer.from_pretrained(
        "DeepPavlov/rubert-base-cased"
    )

    models = {
        "topic": AutoModelForSequenceClassification.from_pretrained(
            f"{path}/topic"
        ),
        "sentiment": AutoModelForSequenceClassification.from_pretrained(
            f"{path}/sentiment"
        ),
        "intent": AutoModelForSequenceClassification.from_pretrained(
            f"{path}/intent"
        ),
    }

    return tokenizer, models



def build_prompt(text, prediction):
    return f"""
Ты — опытный специалист службы поддержки клиентов.
Ниже представлены результаты автоматического анализа обращения.
Тема обращения:
{prediction["topic"]}
Тональность:
{prediction["sentiment"]}
Намерение клиента:
{prediction["intent"]}
Исходное сообщение клиента:
{text}
Твоя задача:

1. Ответить профессионально и вежливо.
2. Учитывать тему, тональность, срочность и намерение клиента.
3. Не придумывать информацию, которой нет.
4. Если данных недостаточно — попросить уточнение.
5. Предложить дальнейшие действия.
6. Ответ должен быть на русском языке.
"""

def generate_answer(text, models, tokenizer):
    from groq import Groq

    API_KEY = "Ваш API"

    client = Groq(api_key=API_KEY)
    prediction = predict_all(text, models, tokenizer)

    prompt = build_prompt(text, prediction)

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.4
    )

    return response.choices[0].message.content

tokenizer, models = load_models(path)

while True:
    from groq import Groq

    API_KEY = "Ваш API ключ"

    client = Groq(api_key=API_KEY)
    text = input("\nВведите обращение (exit для выхода): ")

    if text.lower() == "exit":
        break

    answer = generate_answer(text, models, tokenizer)

    print("\nОтвет:\n")
    print(answer)