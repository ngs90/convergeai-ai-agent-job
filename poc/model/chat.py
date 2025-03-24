
from llama_cpp import Llama
import json 
from jinja2 import Environment, FileSystemLoader



MODEL_PATH = r"C:\Users\bruger-\.lmstudio\models\unsloth\Phi-4-mini-instruct-GGUF\Phi-4-mini-instruct-Q4_K_M.gguf"
# MODEL_PATH = r"C:\Users\bruger-\.lmstudio\models\lmstudio-community\gemma-3-12b-it-GGUF\gemma-3-12b-it-Q4_K_M.gguf"
# MODEL_PATH = r"C:\Users\bruger-\.lmstudio\models\lmstudio-community\gemma-3-1b-it-GGUF\gemma-3-1b-it-Q4_K_M.gguf"
# MODEL_PATH = r"C:\Users\bruger-\.lmstudio\models\lmstudio-community\DeepSeek-R1-Distill-Qwen-7B-GGUF\DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf"
# MODEL_PATH = r"C:\Users\bruger-\.lmstudio\models\second-state\All-MiniLM-L6-v2-Embedding-GGUF\all-MiniLM-L6-v2-Q4_K_S.gguf"
# MODEL_PATH = r"C:\Users\bruger-\.lmstudio\models\kholiavko\intfloat-multilingual-e5-large-instruct\intfloat-multilingual-e5-large.gguf"

class ChatModel:
    def __init__(self, model_path: str, n_ctx=4096*2):
        """Initialize chat model."""
        self.model = Llama(
            model_path=model_path,
            embedding=False, 
            n_ctx=n_ctx,      # Context size, adjust as needed
            n_gpu_layers=-1,  # Use GPU if available (-1 means all layers)
            # chat_format='llama-2'
        )

        # get prompt templates
        env = Environment(loader=FileSystemLoader("poc/prompts"))
        self.job_ad_extract_template = env.get_template("job_ad_extract.jinja").render()
        self.job_metadata_template = env.get_template("job_metadata.jinja").render()

    def extract_job_ad(self, job_ad):

        resp = self.model.create_chat_completion(
            messages = [
                {"role": "system", "content": self.job_ad_extract_template},
                {"role": "user", "content": f"{job_ad}"}
            ],
            response_format={"type": "text"},
            temperature=0)
        return resp["choices"][0]['message']['content']

    def extract_job_ad_metadata(self, job_ad):
        resp = self.model.create_chat_completion(
                messages = [
                    {"role": "system", "content": self.job_metadata_template},
                    {"role": "user", "content": f"""The full document where job ad need to be extracted: \n ---- {job_ad} \n ---- Please return the core job ad descripion"""}
                ],
                response_format={
                    "type": "json_object",
                    "schema": {
                        "type": "object",
                        "properties": {"language": {"type": "string", "max_length": 1},
                                       "job_title": {"type": "string", "max_length": 6},
                                       "company": {"type": "string", "max_length": 5},
                                       "summary": {"type": "string", "max_length": 13, "min_length": 7}
                                       },
                        "required": ["language", 'job_title', 'company', 'summary'],
                    },
                },
                temperature=0,
            )
        
        return json.loads(resp["choices"][0]['message']['content'])

jobad_chat_model = ChatModel(model_path=MODEL_PATH)

if __name__ == '__main__':
    

    with open(r"poc\data\lego.txt") as f:
        job_ad_example = f.read()

    job_ad_cleaned = jobad_chat_model.extract_job_ad(job_ad_example)

    print('RESPONSE\n')
    print(job_ad_cleaned)

    with open(r'poc\data\lego_output.md', 'w') as f:
        f.write(job_ad_cleaned)

    with open(r"poc\data\lego_output.md") as f:
        job_ad_cleaned_l = f.read()
        metadata = jobad_chat_model.extract_job_ad_metadata(job_ad_cleaned_l)

    print('METADATA\n')
    print(metadata, type(metadata))


