import asyncio
import chromadb
from chromadbx import UUIDGenerator
from crawl4ai import AsyncWebCrawler
from bs4 import BeautifulSoup
from poc.model.embedding import jobad_embedding_model
from poc.model.chat import jobad_chat_model
import urllib.parse
import html

def clean_url(url):
    """
    Clean URL by replacing HTML-escaped ampersands
    """
    return html.unescape(url)

async def extract_job_links(search_url):
    """
    Asynchronously extract job links from search page
    """
    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(url=search_url) 
        soup = BeautifulSoup(result.html, "html.parser")
        job_links = [a["href"] for a in soup.select("div.jix_robotjob-inner h4 a")]  # Updated selector
        
        # Normalize links
        job_links = ["https://www.jobindex.dk" + link if link.startswith("/") else link for link in job_links]
        
        return job_links

async def find_next_page(search_url, current_page):
    """
    Find the next page URL in the search results
    
    :param search_url: Base search URL
    :param current_page: Current page number
    :return: Next page URL or None if no more pages
    """
    async with AsyncWebCrawler() as crawler:
        # Add custom headers to mimic a browser request
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
        }
        
        # Modify the crawler run to include custom headers
        result = await crawler.arun(
            url=search_url, 
            headers=headers, 
            wait_for_element='div.jix_pagination'  # Wait for pagination to load
        )
        
        print("Full HTML length:", len(result.html))
        
        soup = BeautifulSoup(result.html, "html.parser")

        with open('soup.txt', 'w') as f:
            f.write(str(search_url))
            f.write(str(soup.prettify()))
        
        pagination_div = soup.find('div', class_='jix_pagination jix_pagination_wide')

        if not pagination_div:
            return None 
        
        # Find all page links within this div
        page_links = pagination_div.select('ul.pagination li.page-item a.page-link')

        print('page_links', page_links)
        
        if page_links:
                next_page_links = [
                    link['href'] for link in page_links 
                    if link.text.strip() == str(current_page + 1)
                ]

                next_page_links = [clean_url(link) for link in next_page_links]
                
                return next_page_links[0] if next_page_links else None
        else:
            return None 
        
async def fetch_job_details(job_url):
    """
    Asynchronously fetch job details for a single URL
    """
    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(url=job_url)
        return result.markdown  # Store raw HTML

class Crawler: 
    def __init__(self, max_pages=5, max_jobs=50):
        """
        Initialize crawler with pagination controls
        
        :param max_pages: Maximum number of search result pages to crawl
        :param max_jobs: Maximum number of job postings to collect
        """
        self.raw_jobs_markdown = []
        self.job_links = []
        self.documents = []
        self.metadatas = []
        self.max_pages = max_pages
        self.max_jobs = max_jobs

    async def crawl(self, keywords):
        """
        Asynchronously crawl job postings with pagination
        """
        # URL encode keywords
        encoded_keywords = urllib.parse.quote(" ".join(keywords))
        base_search_url = f"https://www.jobindex.dk/jobsoegning?q={encoded_keywords}"
        
        current_page = 1
        current_search_url = base_search_url
        
        while current_search_url and len(self.job_links) < self.max_jobs and current_page <= self.max_pages:
            # Extract job links from current page
            page_job_links = await extract_job_links(current_search_url)
            self.job_links.extend(page_job_links)
            
            # Break if we've reached max jobs
            if len(self.job_links) >= self.max_jobs:
                self.job_links = self.job_links[:self.max_jobs]
                break
            
            # Find next page URL
            current_search_url = await find_next_page(base_search_url, current_page)
            current_page += 1
        
        # Concurrently fetch job details
        self.raw_jobs_markdown = await asyncio.gather(
            *[fetch_job_details(link) for link in self.job_links]
        )
        
        return self.job_links, self.raw_jobs_markdown

    def process_job_ad(self):
        """
        Process job advertisements extracted from markdown
        """
        for i, job in enumerate(self.raw_jobs_markdown):
            try:
                processed_job = jobad_chat_model.extract_job_ad(job)
                self.documents.append(processed_job)
            except Exception as e:
                print(f"Error processing job {i}: {e}")
        
        # Determine the language of the cleaned job ad 
        for (job, link) in zip(self.documents, self.job_links):
            try:
                metadata = jobad_chat_model.extract_job_ad_metadata(job) | {'link': link}
                self.metadatas.append(metadata)
            except Exception as e:
                print(f"Error extracting metadata for job: {e}")

        return self.documents, self.metadatas

    def store_in_chroma(self):
        """
        Store processed job postings in ChromaDB
        """
        # Initialize ChromaDB client
        db = chromadb.PersistentClient(path="./chroma_db")
        collection = db.get_or_create_collection(
            name="job_posts",
            embedding_function=jobad_embedding_model
        )
        collection.add(
            ids=UUIDGenerator(len(self.documents)),
            documents=self.documents,
            metadatas=self.metadatas
        )
        
        print(f"Stored {len(self.documents)} job postings in ChromaDB.")

# Only run this if the script is executed directly
if __name__ == '__main__':
    async def main():
        keywords = ['spark', 'python', 'databricks', 'data', 'data science', 'data engineer', 'data scientist', 'Machine learning', 'AI', 'LLM']
        crawler = Crawler(max_pages=5, max_jobs=50)  # Configurable pagination
        await crawler.crawl(keywords=keywords)
        crawler.process_job_ad()
        crawler.store_in_chroma()

    asyncio.run(main())
