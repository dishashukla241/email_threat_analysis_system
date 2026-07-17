import pandas as pd

# ==========================================================
# LOAD DATA
# ==========================================================

df = pd.read_csv("data/processed/emails_final.csv")

# ==========================================================
# TRUSTED DOMAINS
# ==========================================================

trusted_domains = {

    "gmail.com",
    "googlemail.com",
    "outlook.com",
    "hotmail.com",
    "live.com",
    "yahoo.com",
    "ymail.com",
    "icloud.com",
    "me.com",

    "microsoft.com",
    "office.com",
    "apple.com",
    "github.com",
    "gitlab.com",
    "amazon.com",
    "aws.amazon.com",
    "paypal.com",
    "linkedin.com",
    "oracle.com",
    "adobe.com",
    "zoom.us",
    "dropbox.com",
    "proton.me",
    "protonmail.com",

    "aol.com",
    "fastmail.com"

}

# ==========================================================
# DISPOSABLE EMAIL DOMAINS
# ==========================================================

disposable_domains = {

    "mailinator.com",
    "guerrillamail.com",
    "10minutemail.com",
    "temp-mail.org",
    "tempmail.com",
    "throwawaymail.com",
    "fakeinbox.com",
    "trashmail.com"

}

# ==========================================================
# TRUSTED TLDS
# ==========================================================

trusted_tlds = (

    ".edu",
    ".gov",
    ".gov.in",
    ".ac.in",
    ".edu.in"

)

# ==========================================================
# SUSPICIOUS TLDS
# ==========================================================

suspicious_tlds = (

    ".ru",
    ".tk",
    ".xyz",
    ".click",
    ".work",
    ".top",
    ".gq",
    ".zip"

)

# ==========================================================
# REPUTATION FUNCTION
# ==========================================================

def reputation(domain):

    if pd.isna(domain):

        return ("Unknown", 50)

    domain = str(domain).strip().lower()

    # Trusted domain
    if domain in trusted_domains:
        return ("High", 100)

    # Disposable email
    if domain in disposable_domains:
        return ("Very Low", 10)

    # Trusted educational/government
    for tld in trusted_tlds:
        if domain.endswith(tld):
            return ("High", 90)

    # Suspicious TLD
    for tld in suspicious_tlds:
        if domain.endswith(tld):
            return ("Low", 25)

    # Unknown
    return ("Medium", 60)

# ==========================================================
# APPLY
# ==========================================================

rep = df["Sender_Domain"].apply(reputation)

df["Sender_Reputation"] = rep.apply(lambda x: x[0])
df["Sender_Reputation_Score"] = rep.apply(lambda x: x[1])

# ==========================================================
# SAVE
# ==========================================================

df.to_csv(
    "data/processed/emails_reputation.csv",
    index=False
)

print("\nSender Reputation Engine Complete\n")

print(df[[
    "Sender_Domain",
    "Sender_Reputation",
    "Sender_Reputation_Score"
]].head(20))

print("\nDistribution:\n")
print(df["Sender_Reputation"].value_counts())