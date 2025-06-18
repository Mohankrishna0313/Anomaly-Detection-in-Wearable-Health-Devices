import os
import pandas as pd
from flask import Flask, render_template, request, send_file
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import smtplib
from email.mime.text import MIMEText
from dotenv import load_dotenv
from io import StringIO

# Load environment variables
load_dotenv()

EMAIL_USER = os.getenv("EMAIL_USER")
EMAIL_PASS = os.getenv("EMAIL_PASS")
RECIPIENT_EMAIL = os.getenv("RECIPIENT_EMAIL")

app = Flask(__name__)

def load_dataset(file_path):
    """Load dataset from CSV file."""
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return pd.DataFrame()

def detect_anomalies(df):
    """Perform anomaly detection using Isolation Forest."""
    if df.empty:
        return pd.DataFrame()

    df = df.select_dtypes(include=['number'])  # Use only numeric data
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)

    model = IsolationForest(contamination=0.1, random_state=42)
    model.fit(df_scaled)
    df['anomaly'] = model.predict(df_scaled)

    anomalies = df[df['anomaly'] == -1]

    # Add a Serial No column for easy tracking
    anomalies['Serial no'] = anomalies.index + 1

    if not anomalies.empty:
        send_email("🚨 Anomalies Detected", f"Total Anomalies Found: {len(anomalies)}\n{anomalies.to_string(index=False)}")

    return anomalies[['Serial no', 'anomaly']]  # Return only relevant columns

def send_email(subject, body):
    """Send an email notification."""
    if not EMAIL_USER or not EMAIL_PASS or not RECIPIENT_EMAIL:
        print("⚠️ Email configuration is missing. Please check your .env file.")
        return

    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = EMAIL_USER
    msg['To'] = RECIPIENT_EMAIL

    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(EMAIL_USER, EMAIL_PASS)
            server.sendmail(EMAIL_USER, RECIPIENT_EMAIL, msg.as_string())
        print(f"📧 Email alert sent to {RECIPIENT_EMAIL}")
    except Exception as e:
        print(f"❌ Error sending email: {e}")

@app.route("/", methods=["GET", "POST"])
def index():
    anomalies = pd.DataFrame()
    anomalies_count = 0
    file_to_download = None

    if request.method == "POST":
        file = request.files.get("dataset")

        if file:
            file_path = "uploaded_dataset.csv"
            file.save(file_path)
            df = load_dataset(file_path)
            anomalies = detect_anomalies(df)
            anomalies_count = len(anomalies)

            # Save the anomaly results to a CSV file for download
            if not anomalies.empty:
                file_to_download = "anomalies_detected.csv"
                anomalies.to_csv(file_to_download, index=False)

    return render_template("index1.html", anomalies=anomalies, anomalies_count=anomalies_count, file_to_download=file_to_download)

@app.route("/download")
def download():
    file_path = request.args.get("file")
    if file_path and os.path.exists(file_path):
        return send_file(file_path, as_attachment=True)
    return "File not found", 404

if __name__ == "__main__":
    app.run(debug=True)
