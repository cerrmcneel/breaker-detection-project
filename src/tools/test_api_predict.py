import os

import requests


def test_predict_api():
    url = "http://localhost:8000/predict/"
    img_path = "data/dataset/val/images/SPAIN_66e65cb2-f410-4140-86c8-e11439746fc6.jpg"
    
    if not os.path.exists(img_path):
        print("Error: Test image not found.")
        return

    print(f"Sending image {img_path} to {url}...")
    
    with open(img_path, "rb") as f:
        files = {"file": (os.path.basename(img_path), f, "image/jpeg")}
        response = requests.post(url, files=files)

    if response.status_code == 200:
        print("Success!")
        print(response.json())
    else:
        print(f"Error {response.status_code}: {response.text}")

if __name__ == "__main__":
    # Note: The FastAPI server must be running for this to work
    test_predict_api()
