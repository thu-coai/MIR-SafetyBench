# MIR-SafetyBench Evaluation Framework

![Dataset Examples](assets/Data_cases.png)

A comprehensive evaluation framework for assessing multimodal large language models (MLLMs) on multi-image relationship (MIR) based safety attacks.

## 📚 Dataset

**HuggingFace**: [thu-coai/MIR-SafetyBench](https://huggingface.co/datasets/thu-coai/MIR-SafetyBench)  
**Paper**: [arXiv:2601.14127](https://arxiv.org/pdf/2601.14127)

### Dataset Overview

MIR-SafetyBench evaluates MLLM safety through multi-image relationship attacks across 6 safety categories and 9 relationship types:

**Safety Categories:**
- Hate Speech
- Violence
- Self-Harm
- Illegal Activities
- Harassment
- Privacy

**Relationship Types:**
- Analogy
- Causality
- Complementarity
- Decomposition
- Relevance
- Spatial Embedding
- Spatial Juxtaposition
- Temporal Continuity
- Temporal Jump

### Dataset Fields

Each sample contains:
- `id`: Unique identifier
- `original_question`: Original unsafe question
- `relationship_type`: Multi-image relationship type
- `revised_prompt`: Attack prompt utilizing multi-image relationships
- `image_descriptions`: Textual descriptions of images
- `image_keywords`: Keywords for each image
- `images`: List of image file paths
- `iteration`: Generation iteration number

## 🚀 Quick Start

### 1. Download Dataset

Download the dataset from HuggingFace:

```bash
# Set your HuggingFace token
export HUGGINGFACE_TOKEN=your_token_here

# The extraction script will download automatically
# Or download manually to a local path
```

### 2. Extract Dataset

Convert the HuggingFace dataset to local file structure:

```bash
# From HuggingFace (requires HUGGINGFACE_TOKEN)
python extract_data.py

# From local path
python extract_data.py --local-path /path/to/downloaded/dataset

# Specify output directory
python extract_data.py --output ./data --local-path /path/to/dataset
```

This creates a structured directory:
```
output/
├── Hate_Speech/
│   ├── images/
│   │   ├── Analogy/
│   │   │   ├── Analogy_1_0.png
│   │   │   └── ...
│   │   └── ...
│   ├── Analogy_final.json
│   └── ...
├── Violence/
└── ...
```

### 3. Run Evaluation

Evaluate your model on the benchmark:

```bash
# Basic usage
python eval.py \
  --json_dir ./data \
  --models your_model_name \
  --output_dir ./results \
  --model_path /path/to/your/model

# With HarmBench evaluation
python eval.py \
  --json_dir ./data \
  --models your_model_name \
  --output_dir ./results \
  --model_path /path/to/your/model \
  --evaluators harmbench \
  --harmbench_model_path /path/to/HarmBench-Llama-2-13b-cls

# For closed-source models (API)
python eval.py \
  --json_dir ./data \
  --models close_source_model \
  --api_model_name gpt-4o \
  --output_dir ./results
```

## 🔧 Adding Custom Models

Create a new model adapter in the `models/` directory. Your adapter must implement three functions:

```python
def load_model(model_path, num_gpus=1):
    """Load and return model"""
    pass

def infer(pipe, prompts: List[str], image_path_sets: List[List[str]]):
    """Run inference and return results"""
    pass

def unload_model(pipe):
    """Clean up model resources"""
    pass
```

### Model Examples

#### 1. Chat Model (`models/qwen2_5_VL_3B.py`)

For standard vision-language chat models:
- Uses vLLM for efficient inference
- Processes multiple images per prompt
- Returns structured outputs

```bash
python eval.py \
  --json_dir ./data \
  --models qwen2_5_VL_3B \
  --output_dir ./results \
  --model_path /path/to/Qwen2.5-VL-3B-Instruct
```

#### 2. Reasoning Model (`models/GLM-4.1V-9B-Thinking.py`)

For models with chain-of-thought reasoning:
- Extracts answers from reasoning traces
- Handles `<think>` and `<answer>` tags
- Robust parsing for incomplete outputs

```bash
python eval.py \
  --json_dir ./data \
  --models GLM-4.1V-9B-Thinking \
  --output_dir ./results \
  --model_path /path/to/GLM-4.1V-9B-Thinking
```

#### 3. Closed-Source Model (`models/close_source_model.py`)

For API-based models (OpenAI, Claude, etc.):
- Handles rate limiting and retries
- Supports concurrent requests
- Automatic error recovery

```bash
# Set API credentials
export OPENAI_API_KEY=your_api_key

# Run evaluation
python eval.py \
  --json_dir ./data \
  --models close_source_model \
  --api_model_name gpt-4o \
  --output_dir ./results
```

**Environment Variables for API Models:**
```bash
export OPENAI_API_KEY=your_key                    # Required
export CLOSE_SOURCE_API_BASE_URL=https://...      # Optional
export CLOSE_SOURCE_API_NUM_WORKERS=9             # Concurrent processes
export CLOSE_SOURCE_API_TIMEOUT_SEC=149           # Request timeout
export CLOSE_SOURCE_API_MAX_TOTAL_RETRIES=8       # Max retry attempts
```

### Model Configuration

Specify GPU requirements in your model file:

```python
# Single GPU
GPU_REQUIREMENT = 1

# No GPU (for API models)
GPU_REQUIREMENT = 0

# For API models, specify CPU workers
NUM_CPU_WORKERS = 9
```

## 📊 Output Structure

Results are organized by evaluation stage:

```
results/
├── infer/
│   └── {model_name}/
│       └── {category}/
│           └── {relationship_type}.json
└── harmbench/
    └── {model_name}/
        └── {category}/
            └── {relationship_type}.json
```

Each inference result contains:
- `original_question`: Original unsafe question
- `revised_prompt`: Attack prompt with images
- `answer`: Model's response
- `item_index`: Sample index
- `inference_status`: `success`, `failed`, or `crashed`

## 🔍 Evaluation Metrics

The framework supports:
- **HarmBench**: Binary safety classification using HarmBench-Llama-2-13b-cls

Configure evaluators via `--evaluators` flag:

```bash
python eval.py \
  --json_dir ./data \
  --models your_model \
  --output_dir ./results \
  --evaluators harmbench \
  --harmbench_model_path /path/to/HarmBench-Llama-2-13b-cls \
  --harmbench_batch_size 1
```

## 📈 Results Analysis

After evaluation, use `statics.py` to analyze results and compute Attack Success Rate (ASR):

```bash
# Analyze results for a specific model
python statics.py --path ./results/harmbench/{model_name}

# Example
python statics.py --path ./results/harmbench/qwen2_5_VL_3B
```

**Output:**
- Total unsafe count per relationship type
- Total samples per relationship type
- ASR (%) for each relationship type
- Overall ASR across all categories

**Example Output:**
```
==============================================================================
Final Statistics
==============================================================================
Filename                       Unsafe Count  Total Count   ASR(%)    
------------------------------------------------------------------------------
Analogy.json                   45           50           90.00%
Causality.json                 38           45           84.44%
...
------------------------------------------------------------------------------
Total                          423          500          84.60%
==============================================================================
```

The script automatically:
- Traverses all 6 safety category folders
- Aggregates statistics across categories
- Calculates per-relationship-type and overall ASR

## 📋 Requirements

```bash
pip install -r requirements.txt
```

Core dependencies:
- `datasets` - For HuggingFace dataset handling
- `pillow` - For image processing
- `tqdm` - For progress bars
- `torch` - For model inference
- `vllm` - For efficient LLM inference
- `transformers` - For model loading

## 📝 License

See [LICENSE](LICENSE) file for details.

## 📖 Citation

If you use this benchmark, please cite:

```bibtex
@misc{chen2026effectssmartsafetyrisks,
      title={The Side Effects of Being Smart: Safety Risks in MLLMs' Multi-Image Reasoning}, 
      author={Renmiao Chen and Yida Lu and Shiyao Cui and Xuan Ouyang and Victor Shea-Jay Huang and Shumin Zhang and Chengwei Pan and Han Qiu and Minlie Huang},
      year={2026},
      eprint={2601.14127},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2601.14127}, 
}
```