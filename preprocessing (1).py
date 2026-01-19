
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text

def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

fake = pd.read_csv("data/Fake.csv")
true = pd.read_csv("data/True.csv")

fake['label'] = 0
true['label'] = 1

df = pd.concat([fake, true], axis=0).sample(frac=1).reset_index(drop=True)

df['text'] = df['text'].apply(clean_text)
df['text'] = df['text'].apply(preprocess_text)

df.to_csv("data/processed_news.csv", index=False)
print("Preprocessing complete")
