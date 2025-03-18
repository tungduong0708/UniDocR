import requests
from bs4 import BeautifulSoup
import os
import pandas as pd
from urllib.parse import urljoin, urlparse, parse_qs
from tqdm import tqdm  # Import tqdm correctly
import shutil
from PIL import Image


# Global image index
image_counter = 1
image_paths = []  # To store metadata
csv_filename = "queries.csv"

if not os.path.exists(csv_filename):
    pd.DataFrame(columns=["query"]).to_csv(csv_filename, index=False)

def duckduckgo_search(query):
    """Search DuckDuckGo and return the first Wikipedia link."""
    url = f"https://html.duckduckgo.com/html/?q={query}"
    headers = {"User-Agent": "Mozilla/5.0"}
    
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")

    for link in soup.find_all("a", class_="result__a"):
        href = link.get("href")
        if "wiki" in href:  # Choose the first result containing "wiki"
            return fix_url(href)
    
    return None  # No valid link found

def fix_url(duckduckgo_url):
    """Extract the actual URL from a DuckDuckGo redirect."""
    parsed_url = urlparse(duckduckgo_url)
    if parsed_url.scheme == "":
        duckduckgo_url = "https:" + duckduckgo_url  # Add HTTPS if missing

    query_params = parse_qs(parsed_url.query)
    return query_params.get("uddg", [duckduckgo_url])[0]

def download_all_images(page_url, img_path, folder="passage_images"):
    """Download all images from a webpage with global indexing."""
    global image_counter
    headers = {"User-Agent": "Mozilla/5.0"}
    
    response = requests.get(page_url, headers=headers)
    if response.status_code != 200:
        print(f"Failed to access {page_url}")
        return
    
    soup = BeautifulSoup(response.text, "html.parser")
    os.makedirs(folder, exist_ok=True)
    
    images = soup.find_all("img")
    print(f"Found {len(images)} images on {page_url}")

    if len(images) < 3:
        print("Not enough images for ColPali indexing")
        return

    local_paths = []
    for img in images:
        img_url = img.get("src") or img.get("data-src")
        if not img_url:
            continue
        
        img_url = urljoin(page_url, img_url)
        if not img_url.lower().endswith((".jpg", ".jpeg", ".png", ".gif", ".webp")):
            continue

        img_response = requests.get(img_url, headers=headers, stream=True)
        if img_response.status_code == 200:
            img_filename = os.path.join(folder, f"image_{image_counter}.jpg")
            with open(img_filename, "wb") as file:
                for chunk in img_response.iter_content(1024):
                    file.write(chunk)
            # print(f"Downloaded: {img_filename}")
            image_paths.append(img_filename)
            local_paths.append(img_filename)
            image_counter += 1
        else:
            print(f"Failed to download: {img_url}")

    if len(local_paths) >= 3:
        colpali_index_images(local_paths, img_path)

def colpali_index_images(images, img_path):
    global first
    """Apply ColPali indexing to images (second to before last)."""
    if len(images) < 3:
        return
    
    for file_path in images[1:-1]:
        with Image.open(file_path) as img:
            width, height = img.size
            if width <= 100 and height <= 100:
                # print(f"{file_path} is too small (might be logo/symbol)")
                continue
        if not first:
            RAG.index(  # You can load the index first if it exists
                input_path=file_path,  # You can pass the first document here to initialize the index
                index_name="task2",
                store_collection_with_index=False,
                metadata=[image_dict[img_path]],
                overwrite=False
            )
            first = True
        else:
            RAG.add_to_index(  # Use add_to_index instead of index
                file_path,  # Pass each file path separately
                metadata=image_dict[img_path],
                store_collection_with_index=False,  # Whether the index should store the base64 encoded documents
            )
    shutil.rmtree("passage_images")

# Example queries
first = False
for img_path in tqdm(df_passage["page_screenshot"], desc="Processing passages"):
    query = os.path.splitext(os.path.basename(img_path))[0]
    wiki_url = duckduckgo_search(query)
    if wiki_url:
        download_all_images(wiki_url, img_path)
    else:
        print(f"No Wikipedia link found for {img_path}")
    pd.DataFrame({"query": [img_path]}).to_csv(csv_filename, mode="a", header=False, index=False)
    break