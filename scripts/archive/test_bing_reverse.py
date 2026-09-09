import os

import requests


def upload_image(filepath):
    """Uploads a local file to tmpfiles.org and returns the raw download URL."""
    print(f"Uploading {os.path.basename(filepath)} to tmpfiles.org...")
    url = "https://tmpfiles.org/api/v1/upload"
    with open(filepath, 'rb') as f:
        r = requests.post(url, files={'file': f})
        r.raise_for_status()
        data = r.json()
        uploaded_url = data['data']['url']
        # Convert to raw download URL
        raw_url = uploaded_url.replace("tmpfiles.org/", "tmpfiles.org/dl/")
        print(f"Uploaded successfully. Public URL: {raw_url}")
        return raw_url

def query_bing_visual_search(img_url):
    """Queries Bing Visual Search with a public image URL and parses visually similar images."""
    print("Querying Bing Visual Search for visually similar images...")
    bing_url = "https://www.bing.com/images/search"
    params = {
        "view": "detailv2",
        "iss": "sbi",
        "q": f"imgurl:{img_url}"
    }
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5"
    }
    
    r = requests.get(bing_url, params=params, headers=headers, timeout=15)
    r.raise_for_status()
    
    # Let's save the html to inspect it
    with open("bing_search_result.html", "w", encoding="utf-8") as f:
        f.write(r.text)
        
    print("Bing Visual Search page HTML saved to 'bing_search_result.html' for analysis.")

if __name__ == "__main__":
    # Test upload and visual search on one approved image
    approved_dir = "data/scraped_approved"
    files = [os.path.join(approved_dir, f) for f in os.listdir(approved_dir)
             if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
    if files:
        raw_url = upload_image(files[0])
        query_bing_visual_search(raw_url)
    else:
        print("No approved files found.")
