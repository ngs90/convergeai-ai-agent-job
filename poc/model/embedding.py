import chromadb
import numpy as np
from chromadb.api.types import EmbeddingFunction, Documents, Embeddings
from llama_cpp import Llama

# MODEL_PATH = r"C:\Users\bruger-\.lmstudio\models\unsloth\Phi-4-mini-instruct-GGUF\Phi-4-mini-instruct-Q4_K_M.gguf"
# MODEL_PATH = r"C:\Users\bruger-\.lmstudio\models\lmstudio-community\gemma-3-12b-it-GGUF\gemma-3-12b-it-Q4_K_M.gguf"
# MODEL_PATH = r"C:\Users\bruger-\.lmstudio\models\lmstudio-community\gemma-3-1b-it-GGUF\gemma-3-1b-it-Q4_K_M.gguf"
# MODEL_PATH = r"C:\Users\bruger-\.lmstudio\models\lmstudio-community\DeepSeek-R1-Distill-Qwen-7B-GGUF\DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf"
# MODEL_PATH = r"C:\Users\bruger-\.lmstudio\models\second-state\All-MiniLM-L6-v2-Embedding-GGUF\all-MiniLM-L6-v2-Q4_K_S.gguf"
MODEL_PATH = r"C:\Users\bruger-\.lmstudio\models\kholiavko\intfloat-multilingual-e5-large-instruct\intfloat-multilingual-e5-large.gguf"

class JobEmbeddingModel(EmbeddingFunction[Documents]):
    def __init__(self, model_path: str, pooling_strategy='max'):
        """Initialize the embedding model"""

        self.model = Llama(
            model_path=model_path,
            embedding=True,  # Enable embedding mode
            n_ctx=4096*4,      # Context size, adjust as needed
            n_gpu_layers=-1  # Use GPU if available (-1 means all layers)
        )
        self.pooling_strategy = pooling_strategy

    @staticmethod
    def _normalize(embeddings):
        """Normalize embeddings to unit length."""
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / np.maximum(norms, 1e-10)
    
    def __call__(self, input: Documents) -> Embeddings:
        """Generate embeddings for the input documents."""
        embeddings = []
        for text in input:
            # Get embedding for each text
            embedding = self.model.embed(text, normalize=False)

            if isinstance(embedding[0], list):
                if self.pooling_strategy == 'max':
                    embedding = np.max(embedding, axis=0)
                elif self.pooling_strategy == 'mean':
                    embedding = np.mean(embedding, axis=0)
                elif self.pooling_strategy == 'last_hidden_state':
                    embedding = embedding[-1]

            embeddings.append(embedding)

        embeddings_array = np.array(embeddings)
        normalized_embeddings = self._normalize(embeddings_array)
        
        # Convert back to list format for ChromaDB
        return normalized_embeddings.tolist()

# Create embedding function with your model
jobad_embedding_model = JobEmbeddingModel(model_path=MODEL_PATH)


if __name__ == "__main__":
    # Example usage
    documents = [
        "This is a sample document.",
        "Another document for testing.",
        "A third document for embedding. Hurray!"
    ]
    embeddings = jobad_embedding_model(documents)
    print("Embeddings:", embeddings)

    # Initialize ChromaDB with the phi-4 embedding function
    db = chromadb.PersistentClient(path="./chroma_db")
    collection = db.get_or_create_collection(
        name="job_posts_phi4",
        embedding_function=jobad_embedding_model
    )

    # Add stuff 
    # Store in ChromaDB
    collection.add(
        documents=documents,
        ids=[str(x) for x in list(range(0,len(documents)))],
        embeddings=embeddings,
        metadatas=[{'language': 'danish'} for x in range(0,3)]
    )

    # Add more to ChromaDB
    collection.add(
        documents=documents,
        ids=[str(x) for x in list(range(len(documents), 2*len(documents)))],
        metadatas=[{'language': 'danish'} for x in range(0,3)]
    )