
with open("bing_search_result.html", "r", encoding="utf-8") as f:
    html = f.read()

# Let's search for keywords in the entire HTML file
keywords = ["similar", "sbi", "visual", "related", "matches", "result", "grid"]
for kw in keywords:
    count = html.lower().count(kw)
    print(f"Keyword '{kw}' occurs {count} times.")
