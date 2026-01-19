
import pandas as pd

fake_keywords = ['fake', 'hoax', 'false', 'misleading']
real_keywords = ['confirmed', 'official', 'verified', 'statement']

def predict(text):
    fake_score = sum(word in text for word in fake_keywords)
    real_score = sum(word in text for word in real_keywords)
    return 0 if fake_score > real_score else 1

df = pd.read_csv("data/processed_news.csv")
df['prediction'] = df['text'].apply(predict)

accuracy = (df['prediction'] == df['label']).mean()
print("Rule-based Accuracy:", accuracy)
