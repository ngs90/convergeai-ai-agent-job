import asyncio
import chromadb
from chromadbx import UUIDGenerator
from crawl4ai import AsyncWebCrawler
from bs4 import BeautifulSoup
from model.embedding import jobad_embedding_model
from model.chat import jobad_chat_model

async def extract_job_links(search_url):
    """
    Asynchronously extract job links from search page
    """
    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(url=search_url) 
        soup = BeautifulSoup(result.html, "html.parser")
        job_links = [a["href"] for a in soup.select("div.jix_robotjob-inner h4 a")]  # Updated selector
        return ["https://www.jobindex.dk" + link if link.startswith("/") else link for link in job_links]

async def fetch_job_details(job_url):
    """
    Asynchronously fetch job details for a single URL
    """
    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(url=job_url)
        return result.markdown  # Store raw HTML

class Crawler: 
    def __init__(self, n_jobs=5):
        self.raw_jobs_markdown = []
        self.job_links = []
        self.documents = []
        self.metadatas = []
        self.n_jobs = n_jobs

    async def crawl(self, keywords):
        """
        Asynchronously crawl job postings with concurrent fetching
        """
        search_url = "https://www.jobindex.dk/jobsoegning?q=" + "+".join(keywords).replace(' ', '+')
        
        # Extract job links
        self.job_links = await extract_job_links(search_url)

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
            self.documents.append(jobad_chat_model.extract_job_ad(job))
        
        # Determine the language of the cleaned job ad 
        for (job, link) in zip(self.documents, self.job_links):
            self.metadatas.append(jobad_chat_model.extract_job_ad_metadata(job) | {'link': link})


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
        
        print("Job postings stored in ChromaDB.")

# Only run this if the script is executed directly
if __name__ == '__main__':
    async def main():
        keywords = ['spark', 'python', 'databricks']
        crawler = Crawler()
        await crawler.crawl(keywords=keywords)
        crawler.process_job_ad()
        crawler.store_in_chroma()

    asyncio.run(main())