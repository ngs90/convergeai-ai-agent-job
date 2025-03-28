import asyncio
import urllib.parse
from typing import List

from fastapi import FastAPI, Query
from pydantic import BaseModel
import uvicorn

# Import your existing crawler components
from job_crawler.crawler import Crawler

app = FastAPI(
    title="Job Crawler API",
    description="API for crawling job postings with configurable parameters"
)

class JobCrawlerRequest(BaseModel):
    keywords: List[str]
    max_pages: int = 5
    max_jobs: int = 50

class JobCrawlerResponse(BaseModel):
    job_links: List[str]
    job_details: List[str]
    total_jobs_found: int

@app.post("/crawl-jobs/", response_model=JobCrawlerResponse)
async def crawl_jobs(request: JobCrawlerRequest):
    """
    Crawl job postings based on provided keywords, max pages, and max jobs
    
    :param request: Crawling configuration
    :return: Crawled job details
    """
    # Initialize crawler with request parameters
    crawler = Crawler(
        max_pages=request.max_pages, 
        max_jobs=request.max_jobs
    )
    
    # Perform crawling
    job_links, raw_jobs_markdown = await crawler.crawl(request.keywords)
    
    # Process job ads
    documents, metadatas = crawler.process_job_ad()
    
    # Optionally store in ChromaDB
    try:
        crawler.store_in_chroma()
    except Exception as e:
        print(f"Error storing in ChromaDB: {e}")
    
    return JobCrawlerResponse(
        job_links=job_links,
        job_details=documents,
        total_jobs_found=len(job_links)
    )

# Optional: Swagger UI and OpenAPI documentation endpoint
@app.get("/")
async def root():
    return {"message": "Job Crawler API is running"}

# Standalone script to run the server
if __name__ == "__main__":
    uvicorn.run(
        "crawler_api:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=False
    )