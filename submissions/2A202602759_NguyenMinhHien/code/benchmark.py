"""benchmark.py - đo độ trễ suy luận đúng cách (slide Day 2, trang 73 và 75; GUIDE.md mục 4.1).

Đã hoàn thiện từ bộ khung pseudo-code của bài lab.

Quy tắc đo (vi phạm bị trừ điểm, RUBRIC mục 3):
  - warmup: bỏ >= 10 lần chạy đầu
  - đồng bộ GPU: torch.cuda.synchronize() (hoặc CUDA event) TRƯỚC và SAU đoạn cần đo
  - >= 50 lần đo, báo cáo p50, p95, p99 (không chỉ trung bình)
  - ghi rõ GPU, dtype (FP32/AMP/FP16), batch, độ phân giải, có/không gộp BN, phiên bản torch
  - chọn và ghi rõ có tính tiền xử lý hay không
"""
from __future__ import annotations
import time
import numpy as np


def bench(fn, warmup: int = 10, iters: int = 100, sync=None) -> dict:
    """Đo thời gian một hàm `fn()` (không tham số), trả về mili-giây.

    `sync` là hàm đồng bộ (ví dụ torch.cuda.synchronize) hoặc None trên CPU.

    TODO:
      - chạy warmup lần đầu rồi bỏ
      - với mỗi lần đo: sync(); t0 = time.perf_counter(); fn(); sync(); lấy hiệu * 1000
      - trả về {"p50": ..., "p95": ..., "p99": ..., "mean": ..., "n": iters}
    Gợi ý: dùng numpy.percentile hoặc torch.quantile.
    """
    if warmup < 10 or iters < 50:
        raise ValueError("Cần warmup >= 10 và iters >= 50")
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(iters):
        if sync is not None:
            sync()
        start = time.perf_counter()
        fn()
        if sync is not None:
            sync()
        samples.append((time.perf_counter() - start) * 1000)
    p50, p95, p99 = np.percentile(samples, [50, 95, 99])
    return {"p50": float(p50), "p95": float(p95), "p99": float(p99),
            "mean": float(np.mean(samples)), "n": iters}


def latency_report(model, batch_size: int, img_size: int, dtype: str = "fp32", device: str = "cuda",
                   warmup: int = 10, iters: int = 100) -> dict:
    """Đo độ trễ forward của `model` với đầu vào ngẫu nhiên (batch_size, 3, img_size, img_size).

    Trả về dict có thể ghi thẳng vào sheet `Latency` của results.xlsx:
        {"gpu": ..., "dtype": ..., "batch": ..., "img_size": ..., "p50": ..., "p95": ..., "p99": ...,
         "images_per_s": batch_size / (p50 / 1000), "torch": torch.__version__}

    TODO:
      - model.eval(), torch.inference_mode()
      - dtype: "fp32" | "amp" (autocast) | "fp16" (model.half())
      - gọi bench(...) với sync phù hợp; lấy tên GPU bằng torch.cuda.get_device_name
      - Nhớ: ở batch 1, AMP có thể CHẬM hơn FP32 (slide trang 73): đo thật, đừng giả định
    """
    import torch
    if dtype not in ("fp32", "amp", "fp16"):
        raise ValueError("dtype phải là fp32, amp hoặc fp16")
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA chưa sẵn sàng")
    probe = torch.randn(batch_size, 3, img_size, img_size, device=device)
    model = model.to(device).eval()
    if dtype == "fp16":
        if device.type != "cuda":
            raise ValueError("fp16 benchmark chỉ dùng CUDA")
        model = model.half()
        probe = probe.half()
    def forward():
        with torch.inference_mode():
            with torch.autocast(device_type=device.type, enabled=dtype == "amp"):
                model(probe)
    stats = bench(forward, warmup=warmup, iters=iters,
                  sync=torch.cuda.synchronize if device.type == "cuda" else None)
    return {"gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
            "dtype": dtype, "batch": batch_size, "img_size": img_size, **stats,
            "images_per_s": batch_size / (stats["p50"] / 1000),
            "torch": torch.__version__, "includes_preprocessing": False}


def tta_latency(model, k_views: int, **kw) -> dict:
    """Độ trễ của TTA K view: xấp xỉ K lần một lượt chạy (slide trang 63). TODO: đo thật, so với K * p50."""
    import torch
    k_views = int(k_views)
    if k_views < 1:
        raise ValueError("k_views phải >= 1")
    views = kw.get("views")
    if views is None:
        if k_views > 2:
            raise ValueError("K > 2 cần truyền views cụ thể để đo đúng phép TTA")
        views = [lambda x: x] + ([lambda x: torch.flip(x, (-1,))] if k_views == 2 else [])
    if len(views) != k_views:
        raise ValueError("Số views phải bằng k_views")
    device = torch.device(kw.get("device", "cuda"))
    dtype = kw.get("dtype", "fp32")
    batch_size = kw.get("batch_size", 1)
    img_size = kw.get("img_size", 224)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA chưa sẵn sàng")
    model = model.to(device).eval()
    x = torch.randn(batch_size, 3, img_size, img_size, device=device)
    if dtype == "fp16":
        model, x = model.half(), x.half()
    def forward():
        with torch.inference_mode():
            with torch.autocast(device_type=device.type, enabled=dtype == "amp"):
                for view in views:
                    model(view(x))
    stats = bench(forward, warmup=kw.get("warmup", 10), iters=kw.get("iters", 100),
                  sync=torch.cuda.synchronize if device.type == "cuda" else None)
    return {"k_views": k_views, "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
            "dtype": dtype, "batch": batch_size, "img_size": img_size, **stats,
            "images_per_s": batch_size / (stats["p50"] / 1000),
            "torch": torch.__version__, "includes_preprocessing": False}
