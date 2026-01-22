# eval.py
import os
import sys
import json
import pathlib
import importlib.util
import inspect
import multiprocessing
from multiprocessing import Process
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from collections import OrderedDict

from tqdm import tqdm


def _get_torch_num_gpus() -> int:
    import torch
    return torch.cuda.device_count()


def read_json(json_path: str):
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def atomic_write_json(path: str, data: Any):
    tmp = path + ".tmp"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    os.replace(tmp, path)


def load_json_files(json_dir: str):
    files = []
    if not os.path.exists(json_dir):
        print(f"Warning: JSON directory does not exist: {json_dir}")
        return files

    for category_dir in os.listdir(json_dir):
        category_path = os.path.join(json_dir, category_dir)
        if os.path.isdir(category_path):
            for filename in os.listdir(category_path):
                if filename.endswith('_final.json'):
                    files.append(os.path.join(category_path, filename))

    print(f"Found {len(files)} JSON files")
    return files


def build_image_paths(image_paths: List[str], base_dir: str) -> List[str]:
    return [os.path.join(base_dir, p) for p in image_paths]


def extract_category_method(json_path: str) -> Tuple[str, str]:
    try:
        path = pathlib.Path(json_path)
        method = path.stem.replace('_final', '')
        category = path.parent.name
        return category, method
    except Exception as e:
        print(f"Warning: Failed to parse path {json_path}: {e}")
        parts = json_path.split(os.sep)
        if len(parts) >= 2:
            category = parts[-2]
            method = os.path.basename(json_path).replace('_final.json', '')
            return category, method
        raise ValueError(f"Unable to extract category and method from path {json_path}")


def load_model_module(model_name: str):
    model_path = os.path.join(os.path.dirname(__file__), 'models', f'{model_name}.py')
    if not os.path.exists(model_path):
        raise ImportError(f"Model file {model_path} does not exist")

    spec = importlib.util.spec_from_file_location(model_name, model_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load model file {model_path}, spec or loader is None")

    module = importlib.util.module_from_spec(spec)
    sys.modules[model_name] = module
    spec.loader.exec_module(module)

    if not hasattr(module, 'load_model') or not hasattr(module, 'infer') or not hasattr(module, 'unload_model'):
        raise AttributeError(f"Model {model_name} file must implement load_model, infer, and unload_model functions")

    load_sig = inspect.signature(module.load_model)
    infer_sig = inspect.signature(module.infer)

    has_device_param = 'device' in load_sig.parameters
    infer_params = list(infer_sig.parameters.keys())
    if len(infer_params) < 3:
        raise AttributeError(f"Model {model_name} infer function signature should be: infer(pipe, prompts, image_path_sets)")

    # Framework enforces close_source_model GPU_REQUIREMENT=0
    if model_name == "close_source_model":
        gpu_requirement = 0
    else:
        gpu_requirement = getattr(module, 'GPU_REQUIREMENT', 1)

    accepts_preprocessed = getattr(module, 'ACCEPTS_PREPROCESSED_DATA', False)

    print(
        f"Model {model_name} interface verification passed - "
        f"load_model supports device parameter: {has_device_param}, "
        f"GPU requirement: {gpu_requirement}, accepts preprocessed data: {accepts_preprocessed}"
    )
    return module, gpu_requirement, accepts_preprocessed


def check_inference_results_exist(json_files, model_name, output_dir, api_model_name=None):
    existing_files = []
    files_to_process = []

    save_model_name = model_name
    if model_name == 'close_source_model' and api_model_name:
        save_model_name = api_model_name

    for json_path in json_files:
        category, method = extract_category_method(json_path)
        result_dir = os.path.join(output_dir, 'infer', save_model_name, category)
        result_path = os.path.join(result_dir, f'{method}.json')

        source_data = read_json(json_path)
        if isinstance(source_data, dict):
            total_count = 1
            source_items = [source_data]
        else:
            total_count = len(source_data)
            source_items = source_data

        if os.path.exists(result_path):
            try:
                result_data = read_json(result_path)
                if not isinstance(result_data, list):
                    result_data = [result_data]

                completed_indices = set()
                existing_results_by_index = {}

                if result_data and ('item_index' not in result_data[0]):
                    for i, old_item in enumerate(result_data):
                        completed_indices.add(i)
                        existing_results_by_index[i] = old_item
                else:
                    for item in result_data:
                        idx = item.get('item_index', -1)
                        if idx >= 0:
                            completed_indices.add(idx)
                            existing_results_by_index[idx] = item

                missing_indices = set(range(total_count)) - completed_indices

                failed_indices = set()
                for idx, res in existing_results_by_index.items():
                    if res.get('inference_status') in ['failed', 'crashed']:
                        failed_indices.add(idx)

                if not missing_indices and not failed_indices:
                    existing_files.append(result_path)
                    print(f"✓ Inference results complete: {result_path} ({len(completed_indices)}/{total_count})")
                else:
                    files_to_process.append({
                        'json_path': json_path,
                        'result_path': result_path,
                        'completed_indices': completed_indices - failed_indices,
                        'total_count': total_count,
                        'source_items': source_items,
                        'existing_results_by_index': {k: v for k, v in existing_results_by_index.items() if k not in failed_indices}
                    })
                    if failed_indices and missing_indices:
                        print(f"⚠ Inference results incomplete: {result_path} (completed:{len(completed_indices)-len(failed_indices)}/{total_count}, missing:{len(missing_indices)}, failed to retry:{len(failed_indices)})")
                    elif failed_indices:
                        print(f"⚠ Inference results incomplete: {result_path} (completed:{len(completed_indices)-len(failed_indices)}/{total_count}, failed to retry:{len(failed_indices)})")
                    else:
                        print(f"⚠ Inference results incomplete: {result_path} (completed:{len(completed_indices)}/{total_count}, missing:{len(missing_indices)})")

            except Exception as e:
                print(f"⚠ Inference results file corrupted: {result_path}, error: {e}, will regenerate")
                files_to_process.append({
                    'json_path': json_path,
                    'result_path': result_path,
                    'completed_indices': set(),
                    'total_count': total_count,
                    'source_items': source_items,
                    'existing_results_by_index': {}
                })
        else:
            files_to_process.append({
                'json_path': json_path,
                'result_path': result_path,
                'completed_indices': set(),
                'total_count': total_count,
                'source_items': source_items,
                'existing_results_by_index': {}
            })
            print(f"✗ Inference results do not exist: {result_path} (0/{total_count})")

    return existing_files, files_to_process


def producer_process(files_to_process, task_queue, num_consumers, ready_queue, file_info_queue):
    for _ in range(num_consumers):
        ready_queue.get()

    print("Producer: Sending file metadata to Writer...")
    for file_info in files_to_process:
        file_info_queue.put({
            'source_file': file_info['json_path'],
            'total_items': file_info['total_count'],
            'completed_indices': file_info['completed_indices'],
            'existing_results_by_index': file_info['existing_results_by_index']
        })
    file_info_queue.put(None)
    print(f"Producer: File metadata sending complete, total {len(files_to_process)} files")

    print("Producer: Starting task distribution to Workers...")
    for file_info in tqdm(files_to_process, desc="Producer distributing tasks", unit="file"):
        json_path = file_info['json_path']
        completed_indices = file_info['completed_indices']
        total_count = file_info['total_count']
        source_items = file_info['source_items']

        category, method = extract_category_method(json_path)
        base_dir = os.path.dirname(json_path)

        indices_to_process = sorted(list(set(range(total_count)) - completed_indices))
        if not indices_to_process:
            continue

        last_idx_this_round = indices_to_process[-1]

        for idx in indices_to_process:
            item = source_items[idx].copy() if isinstance(source_items[idx], dict) else source_items[idx]
            item['source_file'] = json_path
            item['category'] = category
            item['method'] = method
            item['base_dir'] = base_dir
            item['item_index'] = idx
            item['is_last_item'] = (idx == last_idx_this_round)
            task_queue.put(item)

    for _ in range(num_consumers):
        task_queue.put(None)

    print("Producer: Task distribution complete")


def dynamic_worker(model_name, model_module, pipe, batch_size, task_queue, result_queue, accepts_preprocessed: bool):
    import traceback

    while True:
        batch_items = []
        stop = False

        for _ in range(batch_size):
            item = task_queue.get()
            if item is None:
                stop = True
                break
            batch_items.append(item)

        if not batch_items and stop:
            break
        if not batch_items:
            continue

        try:
            prompts = [it.get('revised_prompt', '') for it in batch_items]
            image_path_sets = [build_image_paths(it.get('images', []), it.get('base_dir', '')) for it in batch_items]

            if accepts_preprocessed:
                preprocess_func = getattr(model_module, 'preprocess_image', None)
                if preprocess_func is not None:
                    image_data_sets = []
                    for img_paths in image_path_sets:
                        preprocessed_imgs = []
                        for img_path in img_paths:
                            if img_path and os.path.exists(img_path):
                                try:
                                    from PIL import Image
                                    img = Image.open(img_path).convert("RGB")
                                    preprocessed_imgs.append(preprocess_func(img))
                                except Exception as e:
                                    print(f"Worker: Unable to load/preprocess image {img_path}: {e}")
                                    preprocessed_imgs.append(None)
                            else:
                                preprocessed_imgs.append(None)
                        image_data_sets.append(preprocessed_imgs)
                else:
                    image_data_sets = image_path_sets
            else:
                image_data_sets = image_path_sets

            try:
                batch_results = model_module.infer(pipe, prompts, image_data_sets)

                # ✅ Critical fix: When plugin returns error, convert it to recognizable failure string
                answers = []
                for res in batch_results:
                    if isinstance(res, dict):
                        if 'output' in res and res['output'] is not None:
                            answers.append(res.get('output', ''))
                        elif 'error' in res:
                            answers.append(f"[INFERENCE_FAILED: {str(res.get('error'))[:200]}]")
                        else:
                            answers.append("[INFERENCE_FAILED: unknown result dict]")
                    else:
                        answers.append(f"[INFERENCE_FAILED: invalid result type {type(res)}]")

            except Exception as e:
                print(f"⚠️ Worker [PID {os.getpid()}]: Batch推理失败: {e}")
                traceback.print_exc()
                answers = [f"[INFERENCE_FAILED: {str(e)[:200]}]" for _ in batch_items]

            for item, ans in zip(batch_items, answers):
                result_queue.put({
                    'original_question': item.get('original_question', ''),
                    'revised_prompt': item.get('revised_prompt', ''),
                    'answer': ans,
                    'source_file': item.get('source_file', ''),
                    'category': item.get('category', ''),
                    'method': item.get('method', ''),
                    'item_index': item.get('item_index', -1),
                    'inference_status': 'failed' if isinstance(ans, str) and ans.startswith('[INFERENCE_FAILED:') else 'success'
                })

        except Exception as e:
            print(f"⚠️ Worker [PID {os.getpid()}]: Batch处理严重错误: {e}")
            traceback.print_exc()
            for item in batch_items:
                result_queue.put({
                    'original_question': item.get('original_question', ''),
                    'revised_prompt': item.get('revised_prompt', ''),
                    'answer': f"[WORKER_CRASHED: {str(e)[:200]}]",
                    'source_file': item.get('source_file', ''),
                    'category': item.get('category', ''),
                    'method': item.get('method', ''),
                    'item_index': item.get('item_index', -1),
                    'inference_status': 'crashed'
                })

        if stop:
            break


def writer_process(result_queue, file_info_queue, output_dir, model_name, api_model_name, total_items, saved_files_queue):
    """
    ✅ Second point: Writer with LRU cache to reduce "clear memory → read disk → clear → read again" IO jitter

    Tunable environment variables:
      WRITER_CACHE_MAX_FILES         default 16
      WRITER_CACHE_MAX_ITEMS_TOTAL   default 50000
      WRITER_INCREMENTAL_THRESHOLD   default 50
    """
    INCREMENTAL_SAVE_THRESHOLD = int(os.environ.get("WRITER_INCREMENTAL_THRESHOLD", "50"))
    CACHE_MAX_FILES = int(os.environ.get("WRITER_CACHE_MAX_FILES", "16"))
    CACHE_MAX_ITEMS_TOTAL = int(os.environ.get("WRITER_CACHE_MAX_ITEMS_TOTAL", "50000"))

    # new results (added in this round)
    results_by_file_index: Dict[str, Dict[int, Dict[str, Any]]] = {}

    # expected/completed indices
    file_expected_counts: Dict[str, int] = {}
    file_completed_indices: Dict[str, set] = {}
    file_last_saved_count: Dict[str, int] = {}
    completed_files: set = set()

    # ✅ LRU cache: cache "saved clean_item (by index)" to avoid frequent disk re-reads
    # key: source_file -> value: dict[index] = clean_item
    cache: "OrderedDict[str, Dict[int, Dict[str, Any]]]" = OrderedDict()
    cache_items_total = 0

    save_model_name = model_name
    if model_name == 'close_source_model' and api_model_name:
        save_model_name = api_model_name

    print("Writer: Reading file information...")
    while True:
        info = file_info_queue.get()
        if info is None:
            break
        sf = info['source_file']
        file_expected_counts[sf] = info['total_items']
        file_completed_indices[sf] = set(info.get('completed_indices', set()))
        file_last_saved_count[sf] = len(file_completed_indices[sf])

        # Put existing results into cache (if any)
        existing = info.get('existing_results_by_index', {}) or {}
        if existing:
            # Convert to clean_item structure to reduce memory
            cleaned = {}
            for idx, item in existing.items():
                cleaned[idx] = {
                    'item_index': idx,
                    'original_question': item.get('original_question', ''),
                    'revised_prompt': item.get('revised_prompt', ''),
                    'answer': item.get('answer', ''),
                }
                if item.get('inference_status') and item['inference_status'] != 'success':
                    cleaned[idx]['inference_status'] = item['inference_status']
            cache[sf] = cleaned
            cache_items_total += len(cleaned)

    print(f"Writer: Preparing to process {len(file_expected_counts)} files")
    print(f"Writer: Incremental save threshold: {INCREMENTAL_SAVE_THRESHOLD}")
    print(f"Writer: LRU cache limit: files={CACHE_MAX_FILES}, items_total={CACHE_MAX_ITEMS_TOTAL}")

    def _cache_touch(sf: str):
        if sf in cache:
            cache.move_to_end(sf, last=True)

    def _cache_evict_if_needed():
        nonlocal cache_items_total
        while (len(cache) > CACHE_MAX_FILES) or (cache_items_total > CACHE_MAX_ITEMS_TOTAL):
            old_sf, old_map = cache.popitem(last=False)
            cache_items_total -= len(old_map)

    def _read_disk_as_cache(sf: str) -> Dict[int, Dict[str, Any]]:
        category, method = extract_category_method(sf)
        result_dir = os.path.join(output_dir, 'infer', save_model_name, category)
        result_path = os.path.join(result_dir, f'{method}.json')
        if not os.path.exists(result_path):
            return {}
        try:
            saved_data = read_json(result_path)
            if not isinstance(saved_data, list):
                saved_data = [saved_data]
            m = {}
            for it in saved_data:
                idx = it.get("item_index", -1)
                if idx >= 0:
                    m[idx] = it
            return m
        except Exception as e:
            print(f"⚠️ Writer: Failed to read back {result_path}: {e}")
            return {}

    def save_file_results(source_file: str, is_final: bool, clear_new_results: bool):
        nonlocal cache_items_total

        category, method = extract_category_method(source_file)
        result_dir = os.path.join(output_dir, 'infer', save_model_name, category)
        result_path = os.path.join(result_dir, f'{method}.json')

        prev_completed_count = len(file_completed_indices.get(source_file, set()))

        # Get from cache first (avoid reading disk)
        if source_file in cache:
            existing_by_index = cache[source_file]
            _cache_touch(source_file)
        else:
            existing_by_index = _read_disk_as_cache(source_file)
            cache[source_file] = existing_by_index
            cache_items_total += len(existing_by_index)
            _cache_evict_if_needed()

        new_by_index = results_by_file_index.get(source_file, {})

        # Merge (new overrides existing)
        all_by_index = dict(existing_by_index)
        all_by_index.update(new_by_index)

        if not all_by_index:
            return

        sorted_indices = sorted(all_by_index.keys())
        clean_results = [all_by_index[i] for i in sorted_indices]

        atomic_write_json(result_path, clean_results)

        # Update status
        file_last_saved_count[source_file] = len(clean_results)
        file_completed_indices[source_file] = set(sorted_indices)

        # ✅ Update cache to latest (still clean_item structure)
        # First update items_total count: subtract old, add new
        if source_file in cache:
            cache_items_total -= len(cache[source_file])
        cache[source_file] = {it["item_index"]: it for it in clean_results}
        cache_items_total += len(cache[source_file])
        _cache_touch(source_file)
        _cache_evict_if_needed()

        # Clear "new results in this round" to save memory (but don't clear cache to avoid re-reading)
        if clear_new_results:
            results_by_file_index.pop(source_file, None)

        if is_final:
            completed_files.add(source_file)
            total = file_expected_counts.get(source_file, "?")
            new_count = len(new_by_index)
            if prev_completed_count > 0:
                print(f"✓ Writer: Completed (resumed) {result_path} (original {prev_completed_count} items + new {new_count} items = {len(clean_results)}/{total})")
            else:
                print(f"✓ Writer: Completed {result_path} ({len(clean_results)}/{total})")

            saved_files_queue.put(result_path)

            # Optionally clear cache for this file after completion to free memory (done here)
            if source_file in cache:
                cache_items_total -= len(cache[source_file])
                cache.pop(source_file, None)

    actual_total = total_items
    with tqdm(total=actual_total, desc="Saving inference results in real-time (index-based)", unit="item") as pbar:
        while True:
            result_item = result_queue.get()
            if result_item is None:
                break

            source_file = result_item.get('source_file', '')
            item_index = result_item.get('item_index', -1)
            if not source_file or item_index < 0:
                print(f"⚠️ Writer: Received invalid result, skipping: {result_item}")
                continue

            results_by_file_index.setdefault(source_file, {})[item_index] = {
                'item_index': item_index,
                'original_question': result_item.get('original_question', ''),
                'revised_prompt': result_item.get('revised_prompt', ''),
                'answer': result_item.get('answer', ''),
                **({'inference_status': result_item.get('inference_status')}
                   if result_item.get('inference_status') and result_item.get('inference_status') != 'success'
                   else {})
            }

            pbar.update(1)

            expected_total = file_expected_counts.get(source_file, -1)
            already_completed = file_completed_indices.get(source_file, set())
            new_completed = set(results_by_file_index.get(source_file, {}).keys())
            all_completed = already_completed | new_completed

            if expected_total > 0 and len(all_completed) >= expected_total:
                save_file_results(source_file, is_final=True, clear_new_results=True)
            else:
                last_saved = file_last_saved_count.get(source_file, len(already_completed))
                current_total = len(all_completed)
                if current_total - last_saved >= INCREMENTAL_SAVE_THRESHOLD:
                    save_file_results(source_file, is_final=False, clear_new_results=True)

    # At end: save incomplete files with data for next resumption
    pending_files = set(results_by_file_index.keys()) - completed_files
    if pending_files:
        print(f"Writer: Found {len(pending_files)} incomplete files, saving checkpoint data...")
        for sf in pending_files:
            save_file_results(sf, is_final=False, clear_new_results=False)
            category, method = extract_category_method(sf)
            result_dir = os.path.join(output_dir, 'infer', save_model_name, category)
            result_path = os.path.join(result_dir, f'{method}.json')
            saved_files_queue.put(result_path)

    saved_files_queue.put(None)
    print("Writer: Completed")


def run_inference_for_model(model_name, batch_size, allocated_gpus, task_queue, result_queue, ready_queue,
                           accepts_preprocessed=False, api_model_name=None, model_path=None):
    if model_name != 'close_source_model':
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([gpu.split(':')[1] for gpu in allocated_gpus])

    model_module, _, _ = load_model_module(model_name)

    try:
        if model_name == 'close_source_model':
            if api_model_name:
                pipe = model_module.load_model(model_name=api_model_name)
            else:
                pipe = model_module.load_model()
        else:
            if model_path:
                pipe = model_module.load_model(model_path=model_path, num_gpus=len(allocated_gpus))
            else:
                pipe = model_module.load_model(num_gpus=len(allocated_gpus))
    except TypeError:
        if model_path:
            try:
                pipe = model_module.load_model(model_path=model_path)
            except TypeError:
                pipe = model_module.load_model()
        else:
            pipe = model_module.load_model()

    if not isinstance(pipe, dict):
        pipe = {'model': pipe}

    ready_queue.put(True)

    dynamic_worker(model_name, model_module, pipe, batch_size, task_queue, result_queue, accepts_preprocessed)

    try:
        model_module.unload_model(pipe)
    except Exception as e:
        print(f"⚠️ unload_model exception occurred (ignored but recommend fixing plugin): {e}")


def run_inference(json_dir, model_list, output_dir, batch_size, api_model_name=None, model_path=None):
    json_files = load_json_files(json_dir)
    if not json_files:
        print(f"Warning: No .json files found in directory {json_dir}.")
        return []

    num_gpus = _get_torch_num_gpus()
    print(f"Detected {num_gpus} available GPUs.")

    model_gpu_requirements = {}
    model_preprocessing_capabilities = {}

    for model_name in model_list:
        try:
            _, gpu_req, accepts_pre = load_model_module(model_name)
            model_gpu_requirements[model_name] = gpu_req
            model_preprocessing_capabilities[model_name] = accepts_pre
        except Exception as e:
            print(f"Warning: Unable to get GPU requirement for model {model_name}, defaulting to 1: {e}")
            model_gpu_requirements[model_name] = 1
            model_preprocessing_capabilities[model_name] = False

    sorted_models = sorted(model_list, key=lambda x: model_gpu_requirements[x], reverse=True)
    print(f"Model processing order: {sorted_models}")

    all_result_files: List[str] = []
    models_to_infer = []

    for model_name in sorted_models:
        print(f"\nChecking inference results for model {model_name}...")
        this_api_model_name = api_model_name if model_name == 'close_source_model' else None
        existing_files, files_to_process = check_inference_results_exist(
            json_files, model_name, output_dir, api_model_name=this_api_model_name
        )
        all_result_files.extend(existing_files)

        if files_to_process:
            models_to_infer.append((model_name, files_to_process))
            new_files = sum(1 for f in files_to_process if len(f['completed_indices']) == 0)
            resume_files = len(files_to_process) - new_files
            if resume_files > 0:
                print(f"Model {model_name} needs to infer {len(files_to_process)} files (new:{new_files}, resume:{resume_files})")
            else:
                print(f"Model {model_name} needs to infer {len(files_to_process)} files")
        else:
            print(f"Model {model_name} all inference results already exist, skipping inference")

    if not models_to_infer:
        print("\nAll models' inference results already exist, no inference needed")
        return all_result_files

    print(f"\nStarting inference, models to infer: {[m for m, _ in models_to_infer]}")

    ctx = multiprocessing.get_context("spawn")

    for model_name, files_to_process in models_to_infer:
        print(f"\nProcessing model: {model_name}")

        gpu_req = model_gpu_requirements[model_name]

        if gpu_req > 0:
            num_processes = num_gpus // gpu_req
        else:
            model_module, _, _ = load_model_module(model_name)
            num_processes = getattr(model_module, 'NUM_CPU_WORKERS', 1)

        if num_processes <= 0:
            print(f"Error: Model {model_name} requires {gpu_req} GPUs, but only {num_gpus} available, skipping.")
            continue

        print(f"Starting {num_processes} processes (GPU requirement per process={gpu_req})")

        queue_size = max(256, batch_size * 64 * max(1, num_processes))

        task_queue = ctx.Queue(maxsize=queue_size)
        result_queue = ctx.Queue(maxsize=queue_size)
        ready_queue = ctx.Queue()
        file_info_queue = ctx.Queue()
        saved_files_queue = ctx.Queue()

        total_items = sum(f['total_count'] - len(f['completed_indices']) for f in files_to_process)
        total_all_items = sum(f['total_count'] for f in files_to_process)
        already_completed = sum(len(f['completed_indices']) for f in files_to_process)
        print(f"Total {total_all_items} items, completed {already_completed} items, pending {total_items} items")

        this_api_model_name = api_model_name if model_name == 'close_source_model' else None

        writer = ctx.Process(
            target=writer_process,
            args=(result_queue, file_info_queue, output_dir, model_name, this_api_model_name, total_items, saved_files_queue),
            daemon=False
        )
        writer.start()

        producer = ctx.Process(
            target=producer_process,
            args=(files_to_process, task_queue, num_processes, ready_queue, file_info_queue),
            daemon=False
        )
        producer.start()

        workers = []
        for i in range(num_processes):
            if gpu_req > 0:
                start_gpu = i * gpu_req
                allocated_gpus = [f"cuda:{j}" for j in range(start_gpu, start_gpu + gpu_req)]
            else:
                allocated_gpus = []

            p = ctx.Process(
                target=run_inference_for_model,
                args=(
                    model_name, batch_size, allocated_gpus, task_queue, result_queue, ready_queue,
                    model_preprocessing_capabilities[model_name], this_api_model_name, model_path
                ),
                daemon=False
            )
            p.start()
            workers.append(p)

        print(f"Started {len(workers)} inference processes + 1 writer process")

        producer.join()
        for p in workers:
            p.join()

        result_queue.put(None)
        writer.join()

        model_result_files = []
        while True:
            x = saved_files_queue.get()
            if x is None:
                break
            if x:
                model_result_files.append(x)

        for rp in model_result_files:
            all_result_files.append(rp)

        seen = set()
        all_result_files = [x for x in all_result_files if not (x in seen or seen.add(x))]

        print(f"Model {model_name} processing complete, added/updated {len(model_result_files)} result files")
        print(f"Total result files: {len(all_result_files)}")

    print("\n" + "=" * 60)
    print(f"All models processing complete, obtained {len(all_result_files)} result files")
    return all_result_files


# ======= Evaluation section: keep original logic (or use my previous corrected version) =======

def run_evaluator_for_process(evaluator_type, result_files, model_path, batch_size, output_dir, allocated_gpus, out_queue):
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([gpu.split(':')[1] for gpu in allocated_gpus])

    judge = None
    try:
        print(f"[PID {os.getpid()}] Loading {evaluator_type} evaluation model... GPUs={allocated_gpus}")

        if evaluator_type == 'harmbench':
            from harmbench_judge import HarmbenchJudge
            judge = HarmbenchJudge(model_path=model_path, batch_size=batch_size, device='auto')
        else:
            raise ValueError(f"Unsupported evaluator type: {evaluator_type}")

        processed_files = []
        for result_path in tqdm(result_files, desc=f"{evaluator_type} evaluation PID {os.getpid()}", unit="file"):
            result_path_obj = Path(result_path)
            infer_root = Path(output_dir) / "infer"
            try:
                rel_path = result_path_obj.relative_to(infer_root)
            except Exception:
                print(f"[PID {os.getpid()}] ⚠️ Path not under infer_root, skipping: {result_path_obj}")
                continue

            if len(rel_path.parts) < 3:
                print(f"[PID {os.getpid()}] ⚠️ rel_path structure abnormal, skipping: {rel_path}")
                continue

            model_name = rel_path.parts[0]
            category = rel_path.parts[1]
            input_filename = rel_path.parts[2]

            output_dir_for_file = Path(output_dir) / evaluator_type / model_name / category
            output_path = output_dir_for_file / input_filename
            output_dir_for_file.mkdir(parents=True, exist_ok=True)

            data = read_json(str(result_path_obj))
            output_data = judge.judge(data)
            atomic_write_json(str(output_path), output_data)

            processed_files.append(str(result_path_obj))

        out_queue.put((evaluator_type, processed_files))

    finally:
        if judge is not None:
            del judge


def run_parallel_evaluation(evaluator_type, result_files, model_path, batch_size, output_dir):
    print(f"\nStarting {evaluator_type} evaluation...")

    if not result_files:
        print(f"No files to evaluate, skipping {evaluator_type}")
        return

    num_gpus = _get_torch_num_gpus()
    print(f"Detected {num_gpus} available GPUs for {evaluator_type} evaluation")

    # gpus_per_process = 2 if evaluator_type == 'harmbench' else 1
    gpus_per_process = 1
    num_processes = min(num_gpus // gpus_per_process, len(result_files))
    if num_processes <= 0:
        print(f"No available GPUs or files, skipping {evaluator_type}")
        return

    ctx = multiprocessing.get_context("spawn")
    out_queue = ctx.Queue()

    print(f"Starting {num_processes} {evaluator_type} evaluation processes, {gpus_per_process} GPUs per process")

    files_per_process = len(result_files) // num_processes
    remainder = len(result_files) % num_processes

    processes = []
    for i in range(num_processes):
        start_idx = i * files_per_process + min(i, remainder)
        end_idx = start_idx + files_per_process + (1 if i < remainder else 0)
        process_files = result_files[start_idx:end_idx]

        allocated_gpus = [f"cuda:{j}" for j in range(i * gpus_per_process, (i + 1) * gpus_per_process)]
        print(f"{evaluator_type} process {i}: GPUs={allocated_gpus}, file count={len(process_files)}")

        p = ctx.Process(
            target=run_evaluator_for_process,
            args=(evaluator_type, process_files, model_path, batch_size, output_dir, allocated_gpus, out_queue),
            daemon=False
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
        if p.exitcode != 0:
            print(f"Subprocess {p.pid} exited abnormally, terminating evaluation")
            for q in processes:
                if q.is_alive():
                    q.terminate()
            raise RuntimeError(f"Evaluation process {p.pid} exited abnormally")

    all_processed_files = []
    for _ in range(num_processes):
        _, processed_files = out_queue.get()
        if processed_files:
            all_processed_files.extend(processed_files)

    print(f"{evaluator_type} evaluation complete, processed {len(all_processed_files)} files")


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)

    import argparse
    parser = argparse.ArgumentParser(description='Batch test multiple models')
    parser.add_argument('--json_dir', type=str, required=True)
    parser.add_argument('--models', type=str, nargs='+', required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=256)

    parser.add_argument('--harmbench_model_path', type=str, default='/WORK/PUBLIC/huangml_work/public_models/HarmBench-Llama-2-13b-cls')
    parser.add_argument('--harmbench_batch_size', type=int, default=1)

    parser.add_argument('--evaluators', type=str, nargs='+', choices=['harmbench'], default=['harmbench'])
    parser.add_argument('--api_model_name', type=str, default=None)
    parser.add_argument('--model_path', type=str, default=None, help='Model file path, passed to model loading function')

    args = parser.parse_args()

    result_files = run_inference(args.json_dir, args.models, args.output_dir, args.batch_size, api_model_name=args.api_model_name, model_path=args.model_path)

    if 'harmbench' in args.evaluators:
        run_parallel_evaluation('harmbench', result_files, args.harmbench_model_path, args.harmbench_batch_size, args.output_dir)


# python eval.py --json_dir 'path/to/data' --models qwen2_5_VL_3B --output_dir path/to/results --harmbench_model_path path/HarmBench-Llama-2-13b-cls --model_path path/Qwen2.5-VL-3B-Instruct