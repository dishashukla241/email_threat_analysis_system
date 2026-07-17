import re
import pandas as pd

# -----------------------------
# Load cleaned dataset
# -----------------------------

df = pd.read_csv("data/processed/emails_cleaned.csv")
df["Body"] = df["Body"].fillna("")

# -----------------------------
# Feature Functions
# -----------------------------

def uppercase_count(text):
    text = str(text)
    return sum(1 for c in text if c.isupper())

def digit_count(text):
    text = str(text)
    return sum(1 for c in text if c.isdigit())

def exclamation_count(text):
    text = str(text)
    return text.count("!")

def question_count(text):
    text = str(text)
    return text.count("?")

def special_character_count(text):
    text = str(text)
    return len(re.findall(r"[^\w\s]", text))

def line_count(text):
    text = str(text)
    return len(text.splitlines())

def average_word_length(text):
    text = str(text)
    words = text.split()

    if len(words) == 0:
        return 0

    return round(sum(len(word) for word in words) / len(words), 2)

# -----------------------------
# Create Features
# -----------------------------

df["Uppercase_Count"] = df["Body"].apply(uppercase_count)

df["Uppercase_Ratio"] = (
    df["Uppercase_Count"] / df["Character_Count"]
).fillna(0)

df["Digit_Count"] = df["Body"].apply(digit_count)

df["Exclamation_Count"] = df["Body"].apply(exclamation_count)

df["Question_Count"] = df["Body"].apply(question_count)

df["Special_Character_Count"] = df["Body"].apply(special_character_count)

df["Line_Count"] = df["Body"].apply(line_count)

df["Average_Word_Length"] = df["Body"].apply(average_word_length)

df["Has_Subject"] = df["Subject"].apply(
    lambda x: 0 if x == "No Subject" else 1
)

df["Has_Links"] = df["URL_Count"].apply(
    lambda x: 1 if x > 0 else 0
)

df["Has_Attachment"] = df["Attachment_Count"].apply(
    lambda x: 1 if x > 0 else 0
)

# -----------------------------
# Save
# -----------------------------

output_path = "data/processed/emails_features.csv"

df.to_csv(output_path, index=False)

# -----------------------------
# Summary
# -----------------------------

print("\nBasic Feature Engineering Complete\n")

print(f"Rows    : {len(df)}")
print(f"Columns : {len(df.columns)}")

print("\nNew Features Added:")

print("""
Uppercase_Count
Uppercase_Ratio
Digit_Count
Exclamation_Count
Question_Count
Special_Character_Count
Line_Count
Average_Word_Length
Has_Subject
Has_Links
Has_Attachment
""")

print(f"\nSaved as: {output_path}")