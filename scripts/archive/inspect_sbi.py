import re

with open("bing_search_result.html", "r", encoding="utf-8") as f:
    html = f.read()

# Let's find occurrences of 'sbi' and print some surrounding context
for idx, match in enumerate(re.finditer(r'sbi', html, re.IGNORECASE)):
    start = max(0, match.start() - 100)
    end = min(len(html), match.end() + 100)
    print(f"Match {idx} (sbi):")
    print(html[start:end].strip())
    print("-" * 50)
    if idx > 10:
        break
