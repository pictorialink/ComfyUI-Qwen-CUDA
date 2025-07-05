from .nodes import *

# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "Qwen3_GGUF": Qwen3_GGUF,
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "Comfyui_Qwen3/Qwen3_GGUF": "Qwen3_GGUF",
}
