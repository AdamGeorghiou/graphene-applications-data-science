from src.collectors.base_collector import BaseCollector
from src.config.settings import PATENTS_DIR
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.common.exceptions import TimeoutException
from datetime import datetime
import time

class GooglePatentCollector(BaseCollector):
    def __init__(self):
        super().__init__(source_name="google_patents", data_dir=PATENTS_DIR)
        
        # Set up Chrome options
        self.options = webdriver.ChromeOptions()
        self.options.add_argument('--headless')
        self.options.add_argument('--no-sandbox')
        self.options.add_argument('--disable-dev-shm-usage')
        self.options.add_argument('--disable-gpu')
        self.options.add_argument('--window-size=1920,1080')
        
    def setup_driver(self):
        """Set up Chrome driver with error handling"""
        try:
            service = Service(ChromeDriverManager().install())
            driver = webdriver.Chrome(service=service, options=self.options)
            return driver
        except Exception as e:
            self.logger.error(f"Error setting up Chrome driver: {str(e)}")
            return None
            
    def search_items(self, num_results=50):
        """Implementation of the abstract method from BaseCollector"""
        self.logger.info(f"Starting search for {num_results} patents")
        driver = None
        
        try:
            driver = self.setup_driver()
            if not driver:
                return
                
            wait = WebDriverWait(driver, 20)
            url = "https://patents.google.com/?q=graphene+applications"
            self.logger.info(f"Navigating to: {url}")
            driver.get(url)
            
            # Wait for the search results to load
            try:
                wait.until(EC.presence_of_element_located((By.TAG_NAME, "article")))
                self.logger.info("Search results loaded successfully")
            except TimeoutException:
                self.logger.error("Timeout waiting for search results to load")
                return
            
            patents_collected = 0
            current_page = 1

            while patents_collected < num_results:
                articles = driver.find_elements(By.TAG_NAME, "article")
                self.logger.info(f"Found {len(articles)} articles on page {current_page}")
                
                for article in articles:
                    if patents_collected >= num_results:
                        break
                    try:
                        title = article.find_element(By.TAG_NAME, "h3").text.strip()
                        
                        link_elem = article.find_element(By.TAG_NAME, "a")
                        link = link_elem.get_attribute("href")
                        patent_id = link.split("/")[-2] if link else "No ID"
                        
                        metadata = article.find_elements(By.TAG_NAME, "h4")
                        metadata_text = metadata[0].text if len(metadata) > 0 else "No metadata"
                        dates_text = metadata[1].text if len(metadata) > 1 else "No dates"
                        
                        abstract = article.find_element(By.CLASS_NAME, "abstract").text.strip()
                        
                        patent_data = {
                            'title': title,
                            'patent_id': patent_id,
                            'link': link,
                            'metadata': metadata_text,
                            'abstract': abstract,
                            'published_date': dates_text,
                            'collection_date': datetime.now().isoformat(),
                            'source': 'Google Patents'
                        }
                        
                        if self.validate_item(patent_data):
                            self.data.append(patent_data)
                            patents_collected += 1
                            self.logger.info(f"Collected patent {patents_collected}/{num_results}: {title[:100]}")
                        
                    except Exception as e:
                        self.logger.error(f"Error processing patent: {str(e)}")
                        continue
                
                if patents_collected >= num_results:
                    break
                
                # Navigate to next page
                try:
                    next_button = driver.find_element(By.XPATH, '//a[@aria-label="Next"]')
                    next_button.click()
                    current_page += 1
                    wait.until(EC.presence_of_element_located((By.TAG_NAME, "article")))
                    time.sleep(2)
                except Exception as e:
                    self.logger.error(f"Error navigating to next page: {str(e)}")
                    break
                    
        except Exception as e:
            self.logger.error(f"Error in search: {str(e)}")
            
        finally:
            if driver:
                driver.quit()

def main():
    collector = GooglePatentCollector()
    collector.search_items(num_results=50)
    df = collector.save_data('google_patents.csv')
    if df is not None:
        analysis = collector.analyze_data(df)
        print("Data collection summary:", analysis)

if __name__ == "__main__":
    main()