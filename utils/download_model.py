import os
import requests

# Configuration
MODEL_URL = "https://github.com/Osetrovie-Podeba/FGD_presentation/releases/download/v0.001/model.keras"
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                          "model", "model.keras")


def download():
    # Create directory
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    # Skip if already exists
    if os.path.exists(MODEL_PATH):
        print(f"✓ Model already exists: {MODEL_PATH}")
        return

    # Download model
    try:
        print("↓ Downloading model...")
        response = requests.get(MODEL_URL)
        response.raise_for_status()

        with open(MODEL_PATH, 'wb') as f:
            f.write(response.content)

        print(f"✓ Model downloaded successfully: {MODEL_PATH}")
        print(f"Set MODEL_PATH={MODEL_PATH} in your environment variables")

    except Exception as e:
        print(f"✗ Error: {str(e)}")
