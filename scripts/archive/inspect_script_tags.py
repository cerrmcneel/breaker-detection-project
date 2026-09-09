import re

with open("bing_search_result.html", "r", encoding="utf-8") as f:
    html = f.read()

# Let's search for JSON data embedded in script tags
scripts = re.findall(r'<script.*?>([\s\S]*?)</script>', html)
print(f"Found {len(scripts)} script tags.")

# Look for patterns like JSON data with 'similar' or 'visual'
found_matches = []
for idx, s in enumerate(scripts):
    if "visuallysimilar" in s.lower() or "similarimages" in s.lower() or "visualsearch" in s.lower() or "vstoken" in s.lower():
        print(f"Script tag {idx} matches visual search query keywords!")
        # Let's extract some text around it
        found_matches.append(s)

# Let's write out script matches to inspect them
for i, m in enumerate(found_matches):
    with open(f"script_match_{i}.txt", "w", encoding="utf-8") as f:
        f.write(m)
    print(f"Saved match to script_match_{i}.txt (size: {len(m)} chars)")
