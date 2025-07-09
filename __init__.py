from .nodes import *

# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "Qwen3": Qwen3,
    "Qwen25_VL": Qwen25VL,
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "Comfyui_Qwen_CUDA/Qwen3": "Qwen3",
    "Comfyui_Qwen_CUDA/Qwen25_VL": "Qwen25_VL",
}
