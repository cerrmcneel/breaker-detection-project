import re

with open("bing_search_result.html", "r", encoding="utf-8") as f:
    html = f.read()

print(f"Total HTML size: {len(html)} characters.")

# Look for image extensions
urls = re.findall(r'https?://[^\s"\'<>]*?\.(?:jpg|jpeg|png|webp)', html, re.IGNORECASE)
print(f"Found {len(urls)} urls with image extensions.")

# Filter out common Bing static/tracking urls
filtered_urls = [u for u in set(urls) if "bing" not in u.lower() and "microsoft" not in u.lower()]
print(f"Found {len(filtered_urls)} non-Bing image URLs.")
for u in list(filtered_urls)[:20]:
    print(" -", u)
