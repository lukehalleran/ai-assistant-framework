"""Quick SMTP connection test — run: python test_smtp.py"""
import smtplib
from dotenv import load_dotenv
import os

load_dotenv()

host = "smtp.gmail.com"
port = 587
user = "lukehalleran@gmail.com"
password = os.getenv("SMTP_PASSWORD", "")

print(f"Connecting to {host}:{port} ...")
try:
    with smtplib.SMTP(host, port, timeout=10) as server:
        server.starttls()
        print("TLS OK")
        server.login(user, password)
        print("Login OK — connection works!")
except smtplib.SMTPAuthenticationError as e:
    print(f"Auth failed: {e}")
except Exception as e:
    print(f"Error: {e}")
