# file: models/glm4v.py

import torch
from vllm import LLM, SamplingParams
from transformers import AutoProcessor
from PIL import Image
from typing import List, Dict, Any
import traceback
import os
import gc
import requests
from io import BytesIO

GPU_REQUIREMENT = 1

def extract_glm_answer(text: str) -> str:
    """
    一个健壮的函数，用于从GLM-4V的输出中提取答案。
    它能处理 <answer> 标签存在但 </answer> 标签缺失的情况。
    """
    # 优先寻找 <answer> 作为起点
    start_marker = "<answer>"
    start_index = text.find(start_marker)

    if start_index != -1:
        # 如果找到了 <answer>，内容从它之后开始
        content_start = start_index + len(start_marker)
        content = text[content_start:]
        
        # 清理掉可能存在的结束标签 </answer>
        end_marker = "</answer>"
        end_index = content.rfind(end_marker) # 用 rfind 从右边找
        if end_index != -1:
            content = content[:end_index]
            
        return content.strip()
    
    # 如果没找到 <answer>，尝试寻找 </think> 作为备选起点
    think_marker = "</think>"
    think_index = text.rfind(think_marker) # 用 rfind 确保找到最后一个
    if think_index != -1:
        # 内容从 </think> 之后开始
        content_start = think_index + len(think_marker)
        return text[content_start:].strip()

    # 如果什么标记都没找到，返回原始文本
    return text.strip()


def load_image(image_source: str) -> Image.Image:
    """
    一个健壮的图片加载函数，同时支持网络URL和本地文件路径。
    这是连接“用户提供路径”和“vLLM需要PIL对象”之间的桥梁。
    """
    if image_source.startswith("http://") or image_source.startswith("https://"):
        response = requests.get(image_source)
        response.raise_for_status()
        return Image.open(BytesIO(response.content)).convert("RGB")
    elif os.path.exists(image_source):
        return Image.open(image_source).convert("RGB")
    else:
        raise FileNotFoundError(f"Image not found at {image_source}")

def load_model(model_path='public_models/GLM-4.1V-9B-Thinking', num_gpus=GPU_REQUIREMENT):
    """
    使用 vLLM 加载 GLM-4.1V 模型
    """
    try:
        print(f"开始使用vLLM加载 GLM-4.1V 模型 (将使用 {num_gpus} 张GPU)...")
        llm = LLM(
            model=model_path,
            tensor_parallel_size=num_gpus,
            trust_remote_code=True,
            gpu_memory_utilization=0.9,
            enforce_eager=True
        )
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        print("GLM-4.1V 模型和 processor 加载完成")
        return {'llm': llm, 'processor': processor}
    except Exception as e:
        print(f"GLM-4.1V 模型加载失败: {e}\n{traceback.format_exc()}")
        raise

def unload_model(pipe: Dict[str, Any]):
    # ... 此函数无变化 ...
    if not pipe: return
    print("开始卸载 GLM-4.1V 模型资源...")
    if 'llm' in pipe: del pipe['llm']
    if 'processor' in pipe: del pipe['processor']
    del pipe
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("GLM-4.1V 模型资源清理完成")


def infer(pipe: Dict[str, Any], prompts: List[str], image_path_sets: List[List[str]]):
    """
    使用 vLLM 的标准批量 API 为 GLM-4.1V 模型进行推理
    """
    llm = pipe['llm']
    processor = pipe['processor']
    
    try:
        if not prompts: return []

        all_inputs = []
        for img_paths, prompt in zip(image_path_sets, prompts):
            # 1. 接受用户传入的路径，并加载成 vLLM 需要的 PIL Image 列表
            images = [load_image(p) for p in img_paths if p]
            
            # 2. 构造一个不含真实图片的 "模板" messages，其唯一目的是生成带 <image> 标记的文本
            content = []
            if images:
                for _ in images:
                    content.append({"type": "image"})
            content.append({"type": "text", "text": prompt})
            messages = [{"role": "user", "content": content}]
            
            # 3. 生成 vLLM 需要的第一个部分：纯文本 prompt
            prompt_text = processor.apply_chat_template(
                messages, 
                add_generation_prompt=True, 
                tokenize=False
            )
            
            # 4. 准备 vLLM 需要的第二个部分：包含真实 PIL 图片的 multi_modal_data
            mm_data = {"image": images} if images else {}
            
            # 5. 将两部分组装起来，形成 vLLM 的标准输入
            all_inputs.append({
                "prompt": prompt_text,
                "multi_modal_data": mm_data
            })

        sampling_params = SamplingParams(
            temperature=0.8, max_tokens=8192, top_p=0.9
        )
        
        outputs = llm.generate(all_inputs, sampling_params)
        
        results = []
        for out in outputs:
            raw_text = out.outputs[0].text
            
            # 使用我们新的、更健壮的函数来提取最终答案
            clean_output = extract_glm_answer(raw_text)
            
            results.append({'output': clean_output})

        return results

    except Exception as e:
        error_msg = str(e) + '\n' + traceback.format_exc()
        return [{'error': error_msg} for _ in prompts]

def main():
    """
    直接运行本文件时的测试入口。
    prompt: "这些图里有什么？"
    images: 三张本地图片路径
    """
    # 请根据实际图片路径修改
    image_paths = [
        'Analogy_0_0_iter1.png',
        'Analogy_0_1_iter1.png',
        'Analogy_1_0_iter1.png',
    ]
    prompt = "Specify in detail what is in each picture respectively"
    try:
        pipe = load_model()
        # 单条prompt配合多张图
        results = infer(pipe, [prompt], [image_paths])
        print("推理结果：")
        for i, res in enumerate(results):
            print(f"样本{i+1}: {res['output'] if 'output' in res else res['error']}")
    finally:
        unload_model(pipe)

if __name__ == "__main__":
    main()
