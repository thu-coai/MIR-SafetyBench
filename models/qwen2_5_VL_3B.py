import torch
from vllm import LLM, SamplingParams
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
from typing import List
import traceback
import os

# GPU需求配置
GPU_REQUIREMENT = 1

def load_model(model_path='public_models/Qwen2.5-VL-3B-Instruct', device='cuda', num_gpus=1):
    """
    使用vLLM加载Qwen2.5-VL-7B-Instruct模型
    返回包含vLLM模型对象和处理器的字典
    """
    try:
        print(f"开始使用vLLM加载Qwen2.5-VL模型到设备: {device}")
        
        # 使用vLLM加载模型
        llm = LLM(
            model=model_path,
            tensor_parallel_size=num_gpus,
            limit_mm_per_prompt={"image": 10, "video": 10},  # 限制每个提示的多模态内容
            trust_remote_code=True,
            dtype="bfloat16" if device.startswith('cuda') else "float32",
        )
        
        # 加载处理器
        print("加载处理器...")
        processor = AutoProcessor.from_pretrained(model_path)
        
        print("vLLM模型和处理器加载完成")
        return {'llm': llm, 'processor': processor}
        
    except Exception as e:
        print(f"vLLM模型加载失败: {e}")
        raise

def unload_model(pipe):
    """
    清理vLLM模型资源
    Args:
        pipe: load_model返回的字典，包含llm和processor对象
    """
    try:
        llm = pipe['llm']
        processor = pipe['processor']
        
        del llm
        del processor
        del pipe
        
        import gc
        gc.collect()
        
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
            
        print("vLLM Qwen2.5-VL模型资源清理完成")
        
    except Exception as e:
        print(f"vLLM模型清理时出错: {e}")



def infer(pipe, prompts: List[str], image_path_sets: List[List[str]]):
    """
    使用vLLM进行批量推理
    Args:
        pipe: 包含llm和processor对象的字典
        prompts: List[str]
        image_path_sets: List[List[str]] - 图片路径列表的列表
    Returns:
        List[dict]: [{'output': ...}, ...]
    """
    llm = pipe['llm']
    processor = pipe['processor']
    
    results = []
    try:
        print(f"开始vLLM推理，批次大小: {len(prompts)}")
        
        # 检查输入
        if not prompts or not image_path_sets:
            print("警告: 输入为空")
            return [{'error': '输入为空'} for _ in prompts]
        
        # 为每个样本构建消息
        all_inputs = []
        for img_paths, prompt in zip(image_path_sets, prompts):
            # 过滤掉空的路径
            valid_paths = [path for path in img_paths if path and os.path.exists(path)]
            
            # 构建消息格式
            messages = [
                {"role": "system", "content": "You are a helpful assistant."}
            ]
            
            content = []
            
            # 添加图像（直接使用文件路径）
            for i, img_path in enumerate(valid_paths):
                content.append({
                    "type": "image", 
                    "image": f"file://{img_path}",
                    "min_pixels": 224 * 224,
                    "max_pixels": 1280 * 28 * 28
                })
            
            # 添加文本
            content.append({"type": "text", "text": prompt})
            
            messages.append({
                "role": "user",
                "content": content
            })
            
            # 使用处理器生成提示
            prompt_text = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            
            # 处理多模态数据
            image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
            
            mm_data = {}
            if image_inputs is not None:
                mm_data["image"] = image_inputs
            if video_inputs is not None:
                mm_data["video"] = video_inputs
            
            llm_inputs = {
                "prompt": prompt_text,
                "multi_modal_data": mm_data,
                "mm_processor_kwargs": video_kwargs,
            }
            
            all_inputs.append(llm_inputs)
        
        print(f"构建了 {len(all_inputs)} 个vLLM输入")
        
        # 设置采样参数
        sampling_params = SamplingParams(
            temperature=0.0,  # 确定性生成
            top_p=0.001,
            repetition_penalty=1.05,
            max_tokens=4096,
            # max_tokens=8192,
            stop_token_ids=[],
        )
        
        # 执行推理
        print("开始生成...")
        outputs = llm.generate(all_inputs, sampling_params=sampling_params)
        
        # 提取结果
        for output in outputs:
            generated_text = output.outputs[0].text.strip()
            results.append({'output': generated_text})
        
        print(f"vLLM生成完成，输出数量: {len(results)}")
        
        # 清理工作完成
        print("推理完成，无需清理临时文件")
        
    except Exception as e:
        error_msg = str(e) + '\n' + traceback.format_exc()
        print(f"vLLM推理过程中出错: {error_msg}")
        for _ in prompts:
            results.append({'error': error_msg})
    
    return results
