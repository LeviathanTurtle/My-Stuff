import requests
from bs4 import BeautifulSoup
import os

url = "https://uesc.io"
response = requests.get(url)
soup = BeautifulSoup(response.text, "html.parser")

# Save HTML
with open("index.html", "w", encoding="utf-8") as f:
    f.write(soup.prettify())

# Download CSS & JS
for tag in soup.find_all(["link", "script"]):
    src = tag.get("href") or tag.get("src")
    if src and not src.startswith("http"):
        file_url = url + src
        file_data = requests.get(file_url).content
        filename = os.path.basename(src)
        with open(filename, "wb") as f:
            f.write(file_data)
