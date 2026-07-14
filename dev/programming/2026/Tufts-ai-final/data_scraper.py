import os
import time
import requests
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By

def download_high_res_images_fallback(search_query, max_images=100):
    folder_name = search_query.replace(" ", "_").lower()
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    print("Initializing Robust Undetectable Chromedriver...")
    options = uc.ChromeOptions()
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    
    driver = uc.Chrome(options=options, use_automation_extension=False)

    # Open target search URL
    search_url = f"https://www.google.com/search?q={search_query}&udm=2"
    driver.get(search_url)
    time.sleep(2)

    image_urls = set()
    thumbnails_processed = 0

    print(f"Gathering images for '{search_query}'...")

    while len(image_urls) < max_images:
        # Dynamically fetch image elements currently loaded on the screen
        thumbnails = driver.find_elements(By.CSS_SELECTOR, "div[data-ri] img, [role='listitem'] img, img")
        
        if not thumbnails or thumbnails_processed >= len(thumbnails):
            # Scroll down to pull the next row of content
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2)
            
            # Click "Show more results" button if it appears
            try:
                show_more = driver.find_element(By.CSS_SELECTOR, "input[type='button'], [role='button']")
                if show_more.is_displayed():
                    show_more.click()
                    time.sleep(1)
            except:
                pass
                
            new_thumbnails = driver.find_elements(By.CSS_SELECTOR, "div[data-ri] img, [role='listitem'] img, img")
            if len(new_thumbnails) == len(thumbnails):
                print("No more thumbnails available on the page.")
                break
            continue

        img = thumbnails[thumbnails_processed]
        thumbnails_processed += 1

        try:
            # Skip layout icons or tracking pixels
            if img.size['width'] < 50 or img.size['height'] < 50:
                continue

            # Open target item view panel
            driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", img)
            img.click()
            time.sleep(1.0) # Lightweight sleep to allow data payload instantiation

            # Extract the actual URL source parameters
            # Google structures their active target layout containers using standard parent/ancestor targets
            parent_elements = driver.find_elements(By.CSS_SELECTOR, "a[href*='imgurl'], div[role='dialog'] a")
            
            for element in parent_elements:
                href = element.get_attribute('href')
                if href and "imgurl=" in href:
                    # Clean out the raw original URL from Google's redirect tracking string
                    raw_url = href.split("imgurl=")[1].split("&")[0]
                    # Decode basic URL hex formatting traits (%3A -> :, %2F -> /)
                    clean_url = requests.utils.unquote(raw_url)
                    
                    if clean_url and clean_url.startswith('http') and clean_url not in image_urls:
                        image_urls.add(clean_url)
                        print(f"Found high-res source link {len(image_urls)}: {clean_url[:60]}...")
                        break
            
        except Exception:
            continue

    driver.quit()

    # Step 2: Download phase
    if not image_urls:
        print("\nCould not resolve target high-res source URLs.")
        return

    print(f"\nDownloading {len(image_urls)} high-resolution source files...")
    for i, url in enumerate(image_urls):
        try:
            headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
            response = requests.get(url, headers=headers, timeout=8)
            
            if response.status_code == 200:
                ext = "jpg"
                if "image/png" in response.headers.get("Content-Type", ""):
                    ext = "png"
                elif "image/webp" in response.headers.get("Content-Type", ""):
                    ext = "webp"

                file_path = os.path.join(folder_name, f"image_{i+1}.{ext}")
                with open(file_path, 'wb') as f:
                    f.write(response.content)
        except Exception:
            continue

    print(f"\nDone! High-res images saved to folder: /{folder_name}")

if __name__ == "__main__":
    query = input("Enter your search query for high-res images: ")
    ammount = input("Enter the maximum number of images to download (default 100): ")
    download_high_res_images_fallback(search_query=query, max_images=int(ammount) if ammount.isdigit() else 100)