import requests
from bs4 import BeautifulSoup
import os
import pandas as pd
from urllib.parse import urljoin, urlparse, parse_qs
from tqdm import tqdm
import shutil
from PIL import Image
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import time
from datasets import load_dataset

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
    fixed_url = query_params.get("uddg", [duckduckgo_url])

    # Ensure fixed_url is always a string
    if isinstance(fixed_url, list):
        fixed_url = fixed_url[0]  # Take first element if it's a list

    return fixed_url

def capture_full_page_screenshot(url, output_path):
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")  # Run in headless mode
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    
    driver.get(url)
    time.sleep(2)  # Wait for page to load

    # Get page dimensions
    total_width = driver.execute_script("return document.body.scrollWidth")
    total_height = driver.execute_script("return document.body.scrollHeight")
    
    # Set window size to capture full page
    driver.set_window_size(total_width, total_height)
    time.sleep(1)  # Allow resize to take effect

    # Take screenshot
    driver.save_screenshot(output_path)
    driver.quit()

# Load dataset
WIT_ds = load_dataset("BByrneLab/multi_task_multi_modal_knowledge_retrieval_benchmark_M2KR", "WIT_passages")
train_ds = WIT_ds['train_passages']

import re

for row in tqdm(train_ds, desc="Processing dataset"):
    wiki_url = row['page_url']  # Extract title-containing field
    passage_id = row["passage_id"]
    
    # if not url:
    #     print("No url found for row, skipping...")
    #     continue

    # print(f"Searching for: {repr(title)}")  # Using repr() to show spaces
    # wiki_url = duckduckgo_search(title)
    
    # print(wiki_url)
    if wiki_url:
        screenshot_path = f"./screenshots/{passage_id}.png"
        os.makedirs("./screenshots", exist_ok=True)
        capture_full_page_screenshot(wiki_url, screenshot_path)
        print(f"Saved screenshot for {passage_id}")
    else:
        print(f"No Wikipedia link found for {passage_id}")
