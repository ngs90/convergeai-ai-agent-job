import chromadb
from model.embedding import jobad_embedding_model
from typing import Optional

# Initialize ChromaDB client

def get_jobs(cv: Optional[str]=None, n_results: int=10, metadata_filters: Optional[dict]=None, must_contain: Optional[str]=None):

    db = chromadb.PersistentClient(path="./chroma_db")
    collection = db.get_collection(name="job_posts", embedding_function=jobad_embedding_model)

    if cv is None:
        resp = collection.peek(n_results)
    else:
        resp = collection.query(query_texts=[cv],
                                n_results=n_results,
                                where=metadata_filters, # format: {"metadata_field": "is_equal_to_this"},
                                where_document={"$contains": must_contain} if must_contain is not None else None
                            )
        
    jobs = []

    if isinstance(resp['ids'][0], list):
        print('no ids', len(resp['ids'][0]))

        for i, id in enumerate(resp['ids'][0]):
            job = {'id': resp['ids'][0][i], 
                'document': resp['documents'][0][i], 
                'langage': resp['metadatas'][0][i]['language'],
                'job_title': resp['metadatas'][0][i]['job_title'],
                'company': resp['metadatas'][0][i]['company'],
                'summary': resp['metadatas'][0][i]['summary'],
                'link': resp['metadatas'][0][i]['link'],
                'distance': resp['distances'][0][i] if cv else None
                }
            jobs.append(job)

    else:
        print('no ids', len(resp['ids']))
        for i, id in enumerate(resp['ids']):
            job = {'id': resp['ids'][i], 
                'document': resp['documents'][i], 
                'langage': resp['metadatas'][i]['language'],
                'job_title': resp['metadatas'][i]['job_title'],
                'company': resp['metadatas'][i]['company'],
                'summary': resp['metadatas'][i]['summary'],
                'link': resp['metadatas'][i]['link'],
                'distance': resp['distances'][i] if cv else None
                }

            jobs.append(job)

    return jobs 


if __name__ == "__main__":
    jobs = get_jobs('I am still studying my bachelor')
    for job in jobs:
        print(job.keys(), job['job_title'], job['distance'])

    jobs = get_jobs()
    for i, job in enumerate(jobs):
        print(job.keys(), job['id'])