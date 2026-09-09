#!/usr/bin/env python3
import hashlib
import os
import random
import re
import sys
import time
import urllib.parse

import requests

# Default search terms targeting Spanish electrical panels
DEFAULT_QUERIES = [
    "cuadro electrico vivienda",
    "cuadro de luces casa",
    "cuadro general de mando y proteccion",
    "magnetotermicos cuadro electrico",
    "diferencial cuadro electrico",
    "spanish breaker panel"
]

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:120.0) Gecko/20100101 Firefox/120.0",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
]

def get_session():
    session = requests.Session()
    session.headers.update({
        "User-Agent": random.choice(USER_AGENTS),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "es-ES,es;q=0.8,en-US;q=0.5,en;q=0.3",
        "Referer": "https://duckduckgo.com/"
    })
    return session

def get_vqd(session, query):
    """Obtains the VQD token required for DuckDuckGo search APIs."""
    url = "https://duckduckgo.com/"
    params = {"q": query}
    try:
        r = session.get(url, params=params, timeout=10)
        r.raise_for_status()
        # Look for vqd=...
        match = re.search(r'vqd=([^\'&"]+)', r.text)
        if match:
            return match.group(1)
        match = re.search(r'vqd\s*[:=]\s*[\'"]?([^\'"&]+)[\'"]?', r.text)
        if match:
            return match.group(1)
    except Exception as e:
        print(f"Error fetching VQD token: {e}")
    return None

def fetch_image_urls(session, query, vqd, max_results=50):
    """Queries DuckDuckGo image JSON API to get direct image URLs."""
    url = "https://duckduckgo.com/i.js"
    image_urls = []
    offset = 0
    
    while len(image_urls) < max_results:
        params = {
            "l": "wt-wt",
            "o": "json",
            "q": query,
            "vqd": vqd,
            "f": ",,,",
            "p": "1",
            "s": str(offset)
        }
        
        try:
            r = session.get(url, params=params, timeout=10)
            r.raise_for_status()
            data = r.json()
            results = data.get("results", [])
            if not results:
                break
            
            for res in results:
                img_url = res.get("image")
                if img_url and img_url not in image_urls:
                    image_urls.append(img_url)
                    if len(image_urls) >= max_results:
                        break
            
            # DuckDuckGo paginates by returned length
            offset += len(results)
            time.sleep(random.uniform(0.5, 1.5))
        except Exception as e:
            print(f"Error fetching image page: {e}")
            break
            
    return image_urls[:max_results]

def sanitize_filename(name):
    """Sanitizes a string to be safe as a filename."""
    return re.sub(r'[^a-zA-Z0-9_-]', '_', name)

def download_images(query, urls, output_dir):
    """Downloads images from the list of URLs into the output directory."""
    os.makedirs(output_dir, exist_ok=True)
    query_slug = sanitize_filename(query).lower()
    
    session = get_session()
    print(f"Downloading images for query: '{query}'...")
    success_count = 0
    
    for idx, url in enumerate(urls):
        # Extract extension or fallback to jpg
        ext = ".jpg"
        parsed = urllib.parse.urlparse(url)
        path = parsed.path
        if "." in path:
            candidate_ext = os.path.splitext(path)[1].lower()
            if candidate_ext in [".jpg", ".jpeg", ".png", ".webp"]:
                ext = candidate_ext
                # Normalize jpeg to jpg
                if ext == ".jpeg":
                    ext = ".jpg"
        
        # Use a stable hash of the URL to prevent duplicates and enable resume
        url_hash = hashlib.md5(url.encode('utf-8')).hexdigest()[:12]
        filename = f"{query_slug}_{url_hash}{ext}"
        filepath = os.path.join(output_dir, filename)
        
        if os.path.exists(filepath):
            print(f"  [{idx+1}/{len(urls)}] Already exists: {filename}")
            success_count += 1
            continue
        
        try:
            r = session.get(url, timeout=8, stream=True)
            r.raise_for_status()
            
            # Check content type is indeed an image
            content_type = r.headers.get("Content-Type", "")
            if "image" not in content_type:
                continue
                
            with open(filepath, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
            success_count += 1
            print(f"  [{idx+1}/{len(urls)}] Downloaded: {filename}")
            
            # Slight delay to be nice
            time.sleep(random.uniform(0.1, 0.4))
        except Exception:
            # Silent fallback or simple log
            pass
            
    print(f"Finished query '{query}'. Successfully processed/downloaded {success_count}/{len(urls)} images.")
    return success_count

def main():
    import argparse
    import json
    parser = argparse.ArgumentParser(description="Image Downloader for Spanish Electrical Panels")
    parser.add_argument("--limit", type=int, default=50, help="Max images to download per query term")
    parser.add_argument("--outdir", type=str, default="data/scraped_raw", help="Output folder path")
    parser.add_argument("--json", type=str, default="data/scraped_urls.json", help="Path to scraped URLs JSON file")
    args = parser.parse_args()
    
    if not os.path.exists(args.json):
        print(f"Error: Scraped URLs file not found at {args.json}")
        print("Please run the browser scraper subagent first to generate the image links.")
        sys.exit(1)
        
    with open(args.json, "r", encoding="utf-8") as f:
        urls_db = json.load(f)
        
    session = get_session()
    total_downloaded = 0
    
    for query, urls in urls_db.items():
        if not urls:
            continue
            
        # Apply download limit per query
        urls_to_download = urls[:args.limit]
        print(f"\nProcessing '{query}' ({len(urls_to_download)} of {len(urls)} URLs)...")
        downloaded = download_images(query, urls_to_download, args.outdir)
        total_downloaded += downloaded
        
        # Throttling delay between query sets
        time.sleep(random.uniform(1.0, 3.0))
        
    print(f"\nDone! Total images downloaded to {args.outdir}: {total_downloaded}")

if __name__ == "__main__":
    main()
