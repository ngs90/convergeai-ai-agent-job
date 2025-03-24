
```
convergeai-ai-agent-job
├─ .python-version
├─ poc
│  ├─ craw4ai.py
│  └─ database.py
├─ pyproject.toml
├─ README.md
└─ uv.lock

```

#### To install llama-cpp-python

$env:FORCE_CMAKE='1'; $env:CMAKE_ARGS='-DGGML_CUDA=on -DLLAMA_AVX=off -DLLAMA_AVX2=off -DLLAMA_FMA=off'
pip install llama-cpp-python --no-cache-dir