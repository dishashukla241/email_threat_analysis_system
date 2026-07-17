import re
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Load Dataset

df = pd.read_csv("data/processed/emails_security.csv")

df["Body"] = df["Body"].fillna("").astype(str)

stop_words = set(ENGLISH_STOP_WORDS)


# Text Preprocessing


def preprocess_text(text):

    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", " ", text)

    # Remove email addresses
    text = re.sub(r"\S+@\S+", " ", text)

    # Remove numbers
    text = re.sub(r"\d+", " ", text)

    # Keep only letters
    text = re.sub(r"[^a-z\s]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    words = text.split()

    # Remove stopwords
    words = [word for word in words if word not in stop_words]

    return " ".join(words)

# Create Processed Text

df["Processed_Text"] = df["Body"].apply(preprocess_text)

# Save

output = "data/processed/emails_final.csv"

df.to_csv(output, index=False)

print("\nNLP Preprocessing Complete\n")

print(f"Rows    : {len(df)}")
print(f"Columns : {len(df.columns)}")

print(f"\nSaved as: {output}")

print("\nExample:\n")

print(df[["Body", "Processed_Text"]].head())