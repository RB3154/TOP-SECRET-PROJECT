import gdown
import os

def download_model():
    model_url = "https://drive.google.com/uc?id=1_4M1nNkJOnyZnucdEPUcwNxI7MC7WJxq"  # Direct ID from your link
    output_path = "best (1).pt"

    if not os.path.exists(output_path):
        print("📥 Downloading model from Google Drive...")
        gdown.download(model_url, output_path, quiet=False)
    else:
        print("✅ Model already downloaded.")

if __name__ == "__main__":
    download_model()
