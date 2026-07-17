import os
import re
import zipfile
import email
import pandas as pd

from bs4 import BeautifulSoup
from email import policy
from tqdm import tqdm

# Paths

ZIP_PATH = "data/raw/archive-2.zip"
EXTRACT_PATH = "data/extracted"
OUTPUT_PATH = "data/processed/emails_master.csv"

# Extract ZIP

if not os.path.exists(EXTRACT_PATH):
    os.makedirs(EXTRACT_PATH)

if len(os.listdir(EXTRACT_PATH)) == 0:
    print("Extracting ZIP...\n")

    with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
        zip_ref.extractall(EXTRACT_PATH)

    print("ZIP extracted successfully.\n")


# Regex

url_pattern = r"https?://\S+|www\.\S+"
email_pattern = r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"

# Function to extract body

def get_body(message):

    body = ""
    is_html = False

    if message.is_multipart():

        for part in message.walk():

            content_type = part.get_content_type()

            if content_type == "text/plain":

                payload = part.get_payload(decode=True)

                if payload:
                    body += payload.decode(errors="ignore")

            elif content_type == "text/html":

                is_html = True

                payload = part.get_payload(decode=True)

                if payload:

                    html = payload.decode(errors="ignore")
                    body += BeautifulSoup(html, "html.parser").get_text()

    else:

        payload = message.get_payload(decode=True)

        if payload:

            body = payload.decode(errors="ignore")

            if message.get_content_type() == "text/html":

                is_html = True
                body = BeautifulSoup(body, "html.parser").get_text()

    return body, is_html


# Read every email


EMAIL_FOLDERS = [
    ("Ham", "data/extracted/archive-2/easy_ham/easy_ham"),
    ("Ham", "data/extracted/archive-2/hard_ham/hard_ham"),
    ("Spam", "data/extracted/archive-2/spam_2/spam_2")
]

rows = []

for label, folder_path in EMAIL_FOLDERS:

    if not os.path.exists(folder_path):
        print(f"Folder not found: {folder_path}")
        continue

    print(f"\nReading {label} emails from {folder_path}")

    for filename in tqdm(os.listdir(folder_path)):

        filepath = os.path.join(folder_path, filename)

        if not os.path.isfile(filepath):
            continue

        try:

            with open(filepath, "rb") as f:

                message = email.message_from_binary_file(
                    f,
                    policy=policy.default
                )

            body, is_html = get_body(message)

            sender = message.get("From")
            receiver = message.get("To")
            subject = message.get("Subject")
            date = message.get("Date")
            cc = message.get("Cc")
            bcc = message.get("Bcc")

            urls = re.findall(url_pattern, body)
            emails = re.findall(email_pattern, body)

            sender_domain = ""

            if sender:

                match = re.search(email_pattern, sender)

                if match:
                    sender_domain = match.group().split("@")[-1]

            attachment_count = 0

            for part in message.walk():

                if part.get_filename():
                    attachment_count += 1

            rows.append({

                "Label": label,
                "File_Name": filename,
                "From": sender,
                "Sender_Domain": sender_domain,
                "To": receiver,
                "CC": cc,
                "BCC": bcc,
                "Subject": subject,
                "Date": date,
                "Body": body,
                "Is_HTML": is_html,
                "URL_Count": len(urls),
                "Email_Count": len(emails),
                "Attachment_Count": attachment_count,
                "Character_Count": len(body),
                "Word_Count": len(body.split())

            })

        except Exception as e:

            print(f"Skipped {filename}: {e}")

# Save CSV


df = pd.DataFrame(rows)

df.to_csv(OUTPUT_PATH, index=False)

print("\n-----------------------------------")
print("Extraction Complete")
print("-----------------------------------")
print(f"Total Emails : {len(df)}")
print(f"Saved To     : {OUTPUT_PATH}")
print("-----------------------------------")