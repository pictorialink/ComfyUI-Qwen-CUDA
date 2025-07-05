import torch
import folder_paths
import os
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer
from llama_cpp import Llama
import re

class Qwen3_GGUF:
    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.model_repo_id = "Qwen/Qwen3-8B-GGUF"
        self.gguf_file = "Qwen3-8B-Q5_K_M.gguf"
        self.model_checkpoint = os.path.join(
            folder_paths.models_dir, "LLM", self.gguf_file
        )
        self.bf16_support = (
            torch.cuda.is_available()
            and torch.cuda.get_device_capability("cuda")[0] >= 8
        )
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "system_prompt": ("STRING", {"default": "你是一个智能助理","multiline": True}),
                "user_prompt": ("STRING", {"multiline": True}),
                "direct": ("BOOLEAN", {"default": False}),
                "keep_model_loaded": ("BOOLEAN", {"default": False}),
                "temperature": (
                    "FLOAT",
                    {"default": 0.7, "min": 0, "max": 1, "step": 0.1},
                ),
                "max_new_tokens": (
                    "INT",
                    {"default": 512, "min": 128, "max": 2048, "step": 1},
                ),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    FUNCTION = "inference"
    CATEGORY = "Comfyui_Qwen3"

    def inference(
        self,
        system_prompt,
        user_prompt,
        direct,
        keep_model_loaded,
        temperature,
        max_new_tokens,
    ):
        if direct:
            return (user_prompt,)
        if not os.path.exists(self.model_checkpoint):
            # 使用 huggingface 下载
            file_path = hf_hub_download(
                repo_id=self.model_repo_id,
                filename=self.gguf_file,
                local_dir=os.path.join(
            folder_paths.models_dir, "LLM")
            )
            print(f"Model downloaded to: {file_path}")
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
        # 3. 格式化提示（Qwen3 使用特定的聊天模板）
        def format_prompt(system_prompt,user_prompt):
            # 禁用 think 模式
            user_prompt += " /no_think"
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            return prompt
        if self.model is None:
            self.model = Llama(
                model_path=self.model_checkpoint,
                n_ctx=4096,  # 上下文长度
                n_gpu_layers=-1,  # -1 表示将所有层卸载到 GPU（如果支持）
                temperature=temperature,  # 控制生成文本的随机性
                max_tokens=max_new_tokens,  # 最大生成 token 数
                verbose=False,
            )
        prompt = format_prompt(system_prompt, user_prompt)
    
        # 5. 生成响应
        def remove_think_tags(text):
            """
            删除文本中所有的<think>标签及其内容，并移除标签后的换行符
            
            参数:
            text (str): 包含XML标签的文本
            
            返回:
            str: 移除<think>标签及后续换行符后的文本
            """
            # 正则表达式模式：匹配<think>标签及其内容，以及紧随其后的换行符
            pattern = r'<think>.*?</think>\s*'
            # 使用re.DOTALL标志使.可以匹配换行符
            # 使用非贪婪匹配(.*?)确保只匹配到最近的</think>
            return re.sub(pattern, '', text, flags=re.DOTALL)
        
        output = self.model(
            prompt,
            max_tokens=2048,
            stop=["</s>"],
            echo=False
        )
        response = remove_think_tags(output["choices"][0]["text"])
        if not keep_model_loaded:
            del self.tokenizer
            del self.model
            self.tokenizer = None
            self.model = None
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        return (response,)
                