import torch
import folder_paths
import os
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer
from llama_cpp import Llama
import re
import json
from pathlib import Path
import gc
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoProcessor,
    BitsAndBytesConfig,
)
from PIL import Image
import numpy as np
from qwen_vl_utils import process_vision_info
import pynvml

def tensor_to_pil(image_tensor, batch_index=0) -> Image:
    # Convert tensor of shape [batch, height, width, channels] at the batch_index to PIL Image
    image_tensor = image_tensor[batch_index].unsqueeze(0)
    i = 255.0 * image_tensor.cpu().numpy()
    img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8).squeeze())
    return img

class ModelsInfo:
    def __init__(self):
        current_dir = Path(__file__).parent.resolve()
        models_info_file = os.path.join(current_dir, "models.json")
        with open(models_info_file, "r", encoding="utf-8") as f:
            self.models_info = json.load(f)

def get_free_vram():
    if torch.cuda.is_available():
        pynvml.nvmlInit()
        # 这里的0是GPU id
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        print(f"总显存: {meminfo.total/1024**3:.2f} GB, 已用显存: {meminfo.used/1024**3:.2f} GB, 剩余显存: {meminfo.free/1024**3:.2f} GB")
        return meminfo.free
    else:
        print("未检测到 GPU")
        return 0

class Qwen25VL(ModelsInfo):
    def __init__(self):
        super().__init__()
        self.model_checkpoint = None
        self.processor = None
        self.model = None
        self.device = "cuda"
        self.bf16_support = (
            torch.cuda.is_available()
            and torch.cuda.get_device_capability(self.device)[0] >= 8
        )

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "text": ("STRING", {"default": "", "multiline": True}),
                "quantization": (
                    ["none", "4bit", "8bit"],
                    {"default": "none"},
                ),
                "temperature": (
                    "FLOAT",
                    {"default": 0.7, "min": 0, "max": 1, "step": 0.1},
                ),
                "max_new_tokens": (
                    "INT",
                    {"default": 512, "min": 128, "max": 2048, "step": 1},
                ),
                "seed": ("INT", {"default": -1}),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "inference"
    CATEGORY = "Comfyui_QwenVL"

    def inference(
        self,
        text,
        quantization,
        temperature,
        max_new_tokens,
        seed,
        image=None,
    ):
        if seed != -1:
            torch.manual_seed(seed)
        # 模型是否存在
        model = self.models_info['cuda'][2]
        self.model_checkpoint = os.path.join(folder_paths.base_path, model['local_path'])
        if not os.path.exists(self.model_checkpoint):
                from huggingface_hub import snapshot_download
                # 使用 huggingface 下载
                file_path = snapshot_download(repo_id=model['repo_id'], local_dir=self.model_checkpoint,local_dir_use_symlinks=False)
                print(f"Model downloaded to: {file_path}")
            
            
        if self.processor is None:
            # Define min_pixels and max_pixels:
            # Images will be resized to maintain their aspect ratio
            # within the range of min_pixels and max_pixels.
            min_pixels = 256*28*28
            max_pixels = 1024*28*28 

            self.processor = AutoProcessor.from_pretrained(
                self.model_checkpoint,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
            )

        if self.model is None:
            # Load the model on the available device(s)
            if quantization == "4bit":
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                )
            elif quantization == "8bit":
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                )
            else:
                quantization_config = None

            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_checkpoint,
                torch_dtype=torch.bfloat16 if self.bf16_support else torch.float16,
                device_map="auto",
                quantization_config=quantization_config,
            )

        with torch.no_grad():
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": text},
                    ],
                }
            ]
            print("deal image")
            pil_image = tensor_to_pil(image)
            messages[0]["content"].insert(0, {
                "type": "image",
                "image": pil_image,
            })

            # 准备输入
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            print("deal messages", messages)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(self.device)

            # 推理
            try:
                generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                result = self.processor.batch_decode(
                    generated_ids_trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                    temperature=temperature,
                )
            except Exception as e:
                return (f"Error during model inference: {str(e)}",)

            del self.processor
            del self.model
            self.processor = None
            self.model = None
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect() 
            return result
    
class Qwen3(ModelsInfo):
    def __init__(self):
        super().__init__()
        self.tokenizer = None
        self.model = None
        gguf_info = self.models_info['cuda'][0]
        tokenizer_info = self.models_info['cuda'][1]
        self.gguf_file = os.path.join(folder_paths.base_path, gguf_info['local_path'], gguf_info['files'][0])
        self.tokenizer_dir = os.path.join(folder_paths.base_path, tokenizer_info['local_path'])
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "system_prompt": ("STRING", {"default": "你是一个智能助理","multiline": True}),
                "user_prompt": ("STRING", {"multiline": True}),
                "direct": ("BOOLEAN", {"default": False}),
                # "keep_model_loaded": ("BOOLEAN", {"default": False}),
                "temperature": (
                    "FLOAT",
                    {"default": 0.7, "min": 0, "max": 1, "step": 0.1},
                ),
                "max_new_tokens": (
                    "INT",
                    {"default": 512, "min": 128, "max": 2048, "step": 1},
                ),
                "seed": ("INT", {
                    "default": 0,  # 默认值
                    "min": 0,      # 最小值
                    "max": 0xffffffffffffffff,  # 最大值（64位整数）
                    "step": 1      # 步长
                }),
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
        # keep_model_loaded,
        temperature,
        max_new_tokens,
        seed=-1
    ):
        if direct:
            return (user_prompt,)
        # 模型是否存在
        models = self.models_info['cuda'][:2]
        for model in models:
            file_dir = os.path.join(folder_paths.base_path, model['local_path'])
            for file_name in model['files']:
                if not os.path.exists(os.path.join(file_dir, file_name)):
                    # 使用 huggingface 下载
                    file_path = hf_hub_download(
                        repo_id=model['repo_id'],
                        filename=file_name,
                        local_dir=file_dir
                    )
                    print(f"Model downloaded to: {file_path}")
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_dir)
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
        # 检查当前显存是否能够加载
        if get_free_vram() < 8 * 1024**3:
            import comfy.model_management as mm
            gc.collect()
            mm.unload_all_models()
            mm.soft_empty_cache()
        if self.model is None:
            self.model = Llama(
                model_path=self.gguf_file,
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
            echo=False,
            seed=seed
        )
        response = remove_think_tags(output["choices"][0]["text"])
        del self.tokenizer
        del self.model
        self.tokenizer = None
        self.model = None
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        return (response,)
                
