# ComfyUI-Qwen3-llama.cpp
Custom nodes for ComfyUI QWen3 8b running based on llama.cpp, which only support the CUDA framework and do not support MPS.

## Installation
Download or git clone this repository into the ComfyUI\custom_nodes\ directory and run:

```bash
pip install -r requirements.txt
```

## About models.json

```json
{
    "cuda": [
        {
            "repo_id": "Qwen/Qwen3-8B-GGUF",
            "local_path": "models/LLM",
            "files": [
                "Qwen3-8B-Q5_K_M.gguf"
            ]
        },
        {
            "repo_id": "Qwen/Qwen3-8B",
            "local_path": "models/LLM",
            "files": [
                "tokenizer_config.json",
                "vocab.json",
                "merges.txt",
                "tokenizer.json",
                "config.json",
                "generation_config.json"
            ]
        }
    ]
}
```

Among them, cuda[0] represents the Qwen3 model. If you want to use models with other parameters or quantization levels, you can modify this part, but note that only Qwen3 can be used. The part of cuda[1] contains the tokenizer configuration information for Qwen3 and does not need to be modified.