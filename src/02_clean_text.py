import re
import pandas as pd

# Load Dataset

df = pd.read_csv("data/processed/emails_master.csv")

# Fill Missing Values


df["From"] = df["From"].fillna("Unknown")

df["Sender_Domain"] = df["Sender_Domain"].fillna("Unknown")

df["Subject"] = df["Subject"].fillna("No Subject")

df["Date"] = df["Date"].fillna("Unknown")

df["CC"] = df["CC"].fillna("")

# Drop BCC because every value is empty
df.drop(columns=["BCC"], inplace=True)

# Clean Email Body


def clean_text(text):

    if pd.isna(text):
        return ""

    # Convert to lowercase
    text = text.lower()

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text)

    # Remove leading/trailing spaces
    text = text.strip()

    return text

df["Body"] = df["Body"].apply(clean_text)

# Recalculate Counts

df["Character_Count"] = df["Body"].str.len()

df["Word_Count"] = df["Body"].str.split().str.len()



output_path = "data/processed/emails_cleaned.csv"

df.to_csv(output_path, index=False)

print("\nCleaning Complete\n")

print("Rows :", len(df))

print("Columns :", len(df.columns))

print("\nSaved as")

print(output_path)