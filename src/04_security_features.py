import re
import pandas as pd

# ------------------------------------
# Load Dataset
# ------------------------------------

df = pd.read_csv("data/processed/emails_features.csv")

df["Body"] = df["Body"].fillna("").astype(str)

# ------------------------------------
# Keyword Lists
# ------------------------------------

SUSPICIOUS_KEYWORDS = [
    "urgent",
    "verify",
    "verification",
    "click",
    "click here",
    "login",
    "password",
    "account",
    "bank",
    "update",
    "confirm",
    "winner",
    "won",
    "lottery",
    "gift",
    "free",
    "claim",
    "limited",
    "offer",
    "bonus",
    "reward",
    "security",
    "payment",
    "invoice",
    "crypto",
    "bitcoin"
]

URGENCY_WORDS = [
    "urgent",
    "immediately",
    "now",
    "today",
    "expire",
    "expired",
    "warning",
    "important",
    "action required"
]

MONEY_WORDS = [
    "bank",
    "payment",
    "credit",
    "debit",
    "refund",
    "money",
    "cash",
    "invoice",
    "bitcoin",
    "crypto",
    "prize"
]

SHORTENED_DOMAINS = [
    "bit.ly",
    "tinyurl",
    "goo.gl",
    "ow.ly",
    "t.co",
    "buff.ly",
    "is.gd"
]

SUSPICIOUS_TLDS = [
    ".ru",
    ".xyz",
    ".top",
    ".click",
    ".work",
    ".gq",
    ".tk"
]

# ------------------------------------
# Helper Functions
# ------------------------------------

def suspicious_keywords(text):

    text = text.lower()

    found = []

    for word in SUSPICIOUS_KEYWORDS:

        if word in text:
            found.append(word)

    return list(set(found))

def contains_words(text, word_list):

    text = text.lower()

    return any(word in text for word in word_list)

# ------------------------------------
# Features
# ------------------------------------

df["Suspicious_Keywords"] = df["Body"].apply(suspicious_keywords)

df["Suspicious_Keyword_Count"] = df["Suspicious_Keywords"].apply(len)

ip_pattern = r"(?:\d{1,3}\.){3}\d{1,3}"

df["IP_Address_Count"] = df["Body"].str.count(ip_pattern)

df["Shortened_URL_Count"] = df["Body"].apply(
    lambda x: sum(domain in x.lower() for domain in SHORTENED_DOMAINS)
)

df["Suspicious_TLD_Count"] = df["Body"].apply(
    lambda x: sum(tld in x.lower() for tld in SUSPICIOUS_TLDS)
)

df["Contains_Urgency"] = df["Body"].apply(
    lambda x: contains_words(x, URGENCY_WORDS)
)

df["Contains_Money"] = df["Body"].apply(
    lambda x: contains_words(x, MONEY_WORDS)
)

# ------------------------------------
# Save
# ------------------------------------

output = "data/processed/emails_security.csv"

df.to_csv(output, index=False)

print("\nSecurity Feature Engineering Complete\n")

print(f"Rows    : {len(df)}")
print(f"Columns : {len(df.columns)}")

print("\nSaved to:")

print(output)