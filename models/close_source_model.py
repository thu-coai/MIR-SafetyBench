import os
import time
import random
import threading
from typing import List, Dict, Any, Optional
from openai import OpenAI, APITimeoutError, APIConnectionError, RateLimitError, APIStatusError

# GPU需求配置（API模型无需本地GPU）
GPU_REQUIREMENT = 0

# API模型期望的CPU进程数（eval.py会用它决定close_source_model启动多少个进程）
NUM_CPU_WORKERS = int(os.environ.get("CLOSE_SOURCE_API_NUM_WORKERS", "9"))

# 声明接受预处理数据：eval.py 会在worker里把PIL Image -> base64字符串
ACCEPTS_PREPROCESSED_DATA = True

# 每个进程内的线程并发（默认1，建议保持1，然后用 NUM_CPU_WORKERS 控总并发）
MAX_CONCURRENT = int(os.environ.get("CLOSE_SOURCE_API_MAX_CONCURRENT", "1"))

# 每个“请求”之间的最小间隔（进程内限速，单位秒）
# 总并发 = NUM_CPU_WORKERS * MAX_CONCURRENT，单个进程内的QPS由这个控制
MIN_INTERVAL_SEC = float(os.environ.get("CLOSE_SOURCE_API_MIN_INTERVAL_SEC", "0"))

# 超时与重试参数
REQUEST_TIMEOUT_SEC = float(os.environ.get("CLOSE_SOURCE_API_TIMEOUT_SEC", "149"))
MAX_TOTAL_RETRIES = int(os.environ.get("CLOSE_SOURCE_API_MAX_TOTAL_RETRIES", "8"))
BASE_BACKOFF_SEC = float(os.environ.get("CLOSE_SOURCE_API_BASE_BACKOFF_SEC", "5"))
MAX_BACKOFF_SEC = float(os.environ.get("CLOSE_SOURCE_API_MAX_BACKOFF_SEC", "120"))
MAX_BLANK_RETRIES = int(os.environ.get("CLOSE_SOURCE_API_MAX_BLANK_RETRIES", "10"))

# 抖动比例：0.2 表示在sleep_time基础上加减最多20%
JITTER_RATIO = float(os.environ.get("CLOSE_SOURCE_API_JITTER_RATIO", "0.2"))


def preprocess_image(pil_image):
    """
    eval.py worker里会调用：把PIL.Image转成base64(无data:前缀)
    """
    import io, base64
    buf = io.BytesIO()
    pil_image.save(buf, format='PNG')
    base64_str = base64.b64encode(buf.getvalue()).decode('utf-8')
    return base64_str


class SimpleRateLimiter:
    """
    进程内限速：保证相邻两次“真实请求”至少间隔 MIN_INTERVAL_SEC
    （跨进程全局限速做起来更复杂；推荐用 NUM_CPU_WORKERS 控总并发）
    """
    def __init__(self, min_interval_sec: float):
        self.min_interval = max(0.0, float(min_interval_sec))
        self._lock = threading.Lock()
        self._next_time = 0.0

    def wait(self):
        if self.min_interval <= 0:
            return
        with self._lock:
            now = time.monotonic()
            if now < self._next_time:
                time.sleep(self._next_time - now)
            self._next_time = time.monotonic() + self.min_interval


def _make_client(base_url: str, api_key: str, timeout_sec: float) -> OpenAI:
    """
    建议关闭SDK内部重试（max_retries=0），避免你自己重试逻辑叠加导致“重试风暴”
    """
    return OpenAI(base_url=base_url, api_key=api_key, timeout=timeout_sec, max_retries=0)


def load_model(
    device=None,
    model_name=None,
    base_url=None,
    api_key=None,
    max_concurrent: Optional[int] = None,
):
    """
    返回 pipe，包含：
      - per-thread client（避免多线程共享client的潜在问题）
      - rate limiter（进程内限速）
    """
    model = model_name or os.environ.get("CLOSE_SOURCE_API_MODEL", "gpt-4o")
    base_url = base_url or os.environ.get("CLOSE_SOURCE_API_BASE_URL", "https://api.openai.com/v1")

    # API key must be provided via environment variable or parameter
    api_key = api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable not set (or pass api_key parameter)")

    max_concurrent = int(max_concurrent) if max_concurrent is not None else MAX_CONCURRENT
    max_concurrent = max(1, max_concurrent)

    pipe = {
        "model": model,
        "base_url": base_url,
        "api_key": api_key,
        "timeout_sec": REQUEST_TIMEOUT_SEC,
        "max_concurrent": max_concurrent,
        "thread_local": threading.local(),
        "limiter": SimpleRateLimiter(MIN_INTERVAL_SEC),
    }
    return pipe


def _get_thread_client(pipe) -> OpenAI:
    tl = pipe["thread_local"]
    if getattr(tl, "client", None) is None:
        tl.client = _make_client(pipe["base_url"], pipe["api_key"], pipe["timeout_sec"])
    return tl.client


def _should_retry_status(status_code: int) -> bool:
    # 429 / 5xx 典型可重试
    if status_code == 429:
        return True
    if 500 <= status_code <= 599:
        return True
    return False


def _sleep_with_jitter(base: float):
    base = max(0.0, base)
    jitter = base * JITTER_RATIO
    if jitter > 0:
        base = base + random.uniform(-jitter, jitter)
    time.sleep(max(0.0, base))


def _single_infer(pipe, prompt: str, image_datas: List[str]) -> Dict[str, Any]:
    """
    单条推理：成功返回 {'output': str}
            失败返回 {'error': str}（infer里会统一转换成 [INFERENCE_FAILED: ...] 输出给框架）
    """
    # content_parts顺序：先图片后文本
    content_parts = []
    for base64_image in image_datas:
        if base64_image is None:
            return {"error": "图片加载失败或未找到"}
        content_parts.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{base64_image}"}
        })
    if prompt:
        content_parts.append({"type": "text", "text": prompt})
    if not content_parts:
        return {"error": "content_parts为空，输入无效"}

    blank_count = 0
    retry_count = 0
    backoff = BASE_BACKOFF_SEC

    while True:
        try:
            # 进程内限速
            pipe["limiter"].wait()

            client = _get_thread_client(pipe)
            response = client.chat.completions.create(
                model=pipe["model"],
                messages=[{"role": "user", "content": content_parts}]
            )
            output = response.choices[0].message.content

            if output and output.strip():
                return {"output": output}

            blank_count += 1
            if blank_count > MAX_BLANK_RETRIES:
                return {"output": ""}

            # 空回复当作可重试（短等）
            _sleep_with_jitter(10)
            continue

        except (APITimeoutError, APIConnectionError, RateLimitError) as e:
            retry_count += 1
            if retry_count > MAX_TOTAL_RETRIES:
                return {"error": f"{type(e).__name__}: {e}"}

            _sleep_with_jitter(backoff)
            backoff = min(backoff * 2, MAX_BACKOFF_SEC)
            continue

        except APIStatusError as e:
            # 有状态码：判断是否可重试
            retry_count += 1
            status = getattr(e, "status_code", None)
            if status is not None and _should_retry_status(int(status)) and retry_count <= MAX_TOTAL_RETRIES:
                _sleep_with_jitter(backoff)
                backoff = min(backoff * 2, MAX_BACKOFF_SEC)
                continue
            return {"error": f"APIStatusError(status={status}): {e}"}

        except Exception as e:
            # 未知异常：有限重试
            retry_count += 1
            if retry_count > MAX_TOTAL_RETRIES:
                return {"error": f"UnknownError: {e}"}
            _sleep_with_jitter(backoff)
            backoff = min(backoff * 2, MAX_BACKOFF_SEC)
            continue


def infer(pipe, prompts: List[str], image_data_sets: List[List[str]]):
    """
    批量推理：每个进程内用线程池做并发（建议 max_concurrent=1）
    返回 List[dict]: [{'output': ...} or {'error': ...}, ...]
    """
    if not prompts or not image_data_sets:
        return [{"error": "输入为空"} for _ in prompts]

    max_concurrent = int(pipe.get("max_concurrent", 1))
    max_concurrent = max(1, max_concurrent)

    results = [{} for _ in range(len(prompts))]

    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrent) as executor:
        future_to_idx = {
            executor.submit(_single_infer, pipe, prompt, image_datas): idx
            for idx, (prompt, image_datas) in enumerate(zip(prompts, image_data_sets))
        }
        for future in concurrent.futures.as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                results[idx] = {"error": f"线程池异常: {e}"}

    return results


def unload_model(pipe):
    # per-thread client 不一定都创建过；无需强制close，尽量安全释放
    # 若你非常希望关闭，可在这里维护一个client列表集中close
    pass