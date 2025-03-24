import streamlit as st
from database import get_jobs
import streamlit as st
import pdfplumber
import docx
import re
import asyncio 
import nest_asyncio
from crawler import Crawler

def extract_text_from_pdf(pdf_file):
    """
    Extract text from a PDF file.
    
    Args:
        pdf_file: Uploaded PDF file
    
    Returns:
        str: Extracted text from the PDF
    """
    try:
        with pdfplumber.open(pdf_file) as pdf:
            text = "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
        return text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
        return None

def extract_text_from_docx(docx_file):
    """
    Extract text from a DOCX file.
    
    Args:
        docx_file: Uploaded DOCX file
    
    Returns:
        str: Extracted text from the DOCX
    """
    try:
        doc = docx.Document(docx_file)
        text = "\n".join([paragraph.text for paragraph in doc.paragraphs if paragraph.text])
        return text
    except Exception as e:
        st.error(f"Error extracting text from DOCX: {e}")
        return None

def extract_text_from_file(uploaded_file):
    """
    Extract text from various file types.
    
    Args:
        uploaded_file: Uploaded file object
    
    Returns:
        str: Extracted text from the file
    """
    if uploaded_file is None:
        return None
    
    # Get file extension
    file_extension = uploaded_file.name.split('.')[-1].lower()
    
    # Read text for plain text/markdown files
    if file_extension in ['txt', 'md']:
        return uploaded_file.getvalue().decode('utf-8')
    
    # PDF handling
    elif file_extension == 'pdf':
        return extract_text_from_pdf(uploaded_file)
    
    # DOCX handling
    elif file_extension == 'docx':
        return extract_text_from_docx(uploaded_file)
    
    else:
        st.error(f"Unsupported file type: {file_extension}")
        return None

def process_keywords(keywords_input):
    """
    Process keywords input by stripping special characters and whitespace.
    
    Args:
        keywords_input (str): Comma-separated keywords
    
    Returns:
        list: Processed keywords
    """
    # Split by comma and process each keyword
    keywords = [
        re.sub(r'[^\w\s]', '', keyword.strip()).lower() 
        for keyword in keywords_input.split(',') 
        if keyword.strip()
    ]
    return keywords

# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()

def run_async_crawler(keywords):
    async def async_crawl():
        # Create crawler instance
        crawler = Crawler()
        
        try:
            # Update progress bar
            my_bar = st.progress(0, text="Finding new relevant jobs. Please wait.")
            
            # Run crawl method
            job_links, raw_markdown = await crawler.crawl(keywords=keywords)
            
            # Update progress bar
            my_bar.progress(20, text="Extracting job ads and metadata. Please wait.")
            
            # Process job ads
            documents, metadatas = crawler.process_job_ad()
            
            # Update progress bar
            my_bar.progress(60, text="Making ready to analyze")
            
            # Store in ChromaDB
            crawler.store_in_chroma()
            
            # Update progress bar
            my_bar.progress(80, text="Finding most relevant jobs.")
            
            # Get jobs
            jobs = get_jobs(keywords, n_results=6)
            
            # Update progress bar
            my_bar.progress(100, text="Ready.")
            
            return jobs
        
        except Exception as e:
            st.error(f"An error occurred during crawling: {e}")
            return []

    # Use asyncio.run with the async function
    return asyncio.run(async_crawl())


def main():
    """
    Main Streamlit application for CV, Cover Letter, and Keywords Upload
    """
    st.title("Job Application Document Processor")
    
    # Initialize session state variables
    if 'cv_document' not in st.session_state:
        st.session_state.cv_document = None
    if 'cl_document' not in st.session_state:
        st.session_state.cl_document = None
    if 'job_keywords' not in st.session_state:
        st.session_state.job_keywords = []
    
    # Sidebar for configuration
    st.sidebar.header("Document Upload")
    
    # CV Upload
    st.sidebar.subheader("CV Upload")
    cv_file = st.sidebar.file_uploader(
        "Upload CV", 
        type=['pdf', 'docx', 'txt', 'md'], 
        help="Upload your CV in PDF, DOCX, TXT, or MD format"
    )
    
    # Cover Letter Upload
    st.sidebar.subheader("Cover Letter Upload")
    cover_letter_file = st.sidebar.file_uploader(
        "Upload Cover Letter", 
        type=['pdf', 'docx', 'txt', 'md'], 
        help="Upload your cover letter in PDF, DOCX, TXT, or MD format"
    )
    
    # Keywords Input
    st.sidebar.subheader("Job Keywords")
    keywords_input = st.sidebar.text_input(
        "Enter keywords (comma-separated)", 
        help="Enter job-related keywords separated by commas"
    )
    
    # Process Documents and Keywords
    if st.sidebar.button("Process Documents"):
        # Extract CV text
        st.session_state.cv_document = extract_text_from_file(cv_file) if cv_file else None
        
        # Extract Cover Letter text
        st.session_state.cl_document = extract_text_from_file(cover_letter_file) if cover_letter_file else None
        
        # Process Keywords
        st.session_state.job_keywords = process_keywords(keywords_input) if keywords_input else []
        
        # Display processing results
        st.success("Documents and keywords processed successfully!")
    
    # Display Processed Information
    st.header("Processed Information")
    
    # CV Preview
    if st.session_state.cv_document:
        st.subheader("CV Content")
        st.text_area("CV Text", st.session_state.cv_document, height=300)
    
    # Cover Letter Preview
    if st.session_state.cl_document:
        st.subheader("Cover Letter Content")
        st.text_area("Cover Letter Text", st.session_state.cl_document, height=300)
    
    # Keywords Preview
    if st.session_state.job_keywords:
        st.subheader("Processed Keywords")
        st.write(st.session_state.job_keywords)

    # In your Streamlit code
    if st.session_state.job_keywords:
        jobs = run_async_crawler(st.session_state.job_keywords)
        st.text_area("Job ads:", str(jobs), height=300)



if __name__ == "__main__":
    main()



# # Function to show job cards
# def show_jobs():
#     st.title("📄 Job Listings")
#     st.write("Click on a job to see full details.")

#     # Display each job card
#     for job in jobs:
#         st.markdown(
#             f"""
#             <div class="job-card">
#                 <h3>{job["job_title"]}</h3>
#                 <h4>{job["company"]}</h4>
#                 <p>{job["summary"]}</p>
#                 <a href="?job_id={job['id']}" class="job-button">View Details</a>
#             </div>
#             """,
#             unsafe_allow_html=True,
#         )
    
#     # Close the job grid
#     st.markdown('</div>', unsafe_allow_html=True)

# # Function to show job details
# def show_job_detail(job_id):
#     job = next((j for j in jobs if j["id"] == job_id), None)
#     if not job:
#         st.error("Job not found!")
#         return

#     # Add custom CSS for job detail page
#     st.markdown(
#         """
#         <style>
#             .back-button {
#                 display: inline-block;
#                 padding: 8px 16px;
#                 background-color: #f1f1f1;
#                 color: #333;
#                 text-decoration: none;
#                 border-radius: 4px;
#                 margin-bottom: 20px;
#                 font-weight: 500;
#                 border: 1px solid #ddd;
#             }
#             .back-button:hover {
#                 background-color: #e0e0e0;
#             }
#             .job-header {
#                 margin-bottom: 30px;
#             }
#             .job-header h1 {
#                 color: #333333;
#             }
#             .job-header h3 {
#                 color: #555555;
#             }
#             .apply-button {
#                 display: inline-block;
#                 padding: 10px 20px;
#                 background-color: #2e7d32;
#                 color: white;
#                 text-decoration: none;
#                 border-radius: 5px;
#                 font-weight: 500;
#                 margin-top: 20px;
#             }
#             .apply-button:hover {
#                 background-color: #1b5e20;
#             }
#             .job-content {
#                 line-height: 1.6;
#                 margin-top: 20px;
#                 color: #333333;
#                 background-color: #ffffff;
#                 padding: 20px;
#                 border-radius: 5px;
#                 border: 1px solid #e0e0e0;
#             }
#         </style>
#         """,
#         unsafe_allow_html=True
#     )

#     # Back button at the top
#     st.markdown('<a href="?home=true" class="back-button">← Back to Job Listings</a>', unsafe_allow_html=True)
    
#     # Job header info
#     st.markdown(f'<div class="job-header"><h1>{job["job_title"]}</h1><h3>{job["company"]}</h3></div>', unsafe_allow_html=True)
    
#     # Job content - using st.markdown to properly render the markdown content
#     st.markdown('<div class="job-content">', unsafe_allow_html=True)
#     st.markdown(job["document"])
#     st.markdown('</div>', unsafe_allow_html=True)
    
#     # Apply button
#     st.markdown(f'<a href="{job["link"]}" target="_blank" class="apply-button">Apply for this position</a>', unsafe_allow_html=True)

# # Routing logic
# query_params = st.query_params
# if "job_id" in query_params:
#     show_job_detail(query_params["job_id"])
# else:
#     show_jobs()