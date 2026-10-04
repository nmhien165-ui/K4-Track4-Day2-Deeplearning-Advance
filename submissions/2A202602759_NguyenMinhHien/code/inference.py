"""inference.py - các phương pháp suy luận (Bước 3 của GUIDE.md).

Đã hoàn thiện từ bộ khung pseudo-code của bài lab.
Liên hệ slide Day 2: TTA (trang 62-66, 75), ensemble/EMA/soup (trang 67), độ phân giải kiểm tra
(trang 68), temperature scaling (trang 69), gộp BatchNorm (trang 71).

Mọi hàm phải chạy ở chế độ eval, không gradient. Chọn phương pháp CHỈ dựa trên val;
nhiệt độ T khớp trên VAL rồi áp dụng sang test (README.md, S2 và S4).

Giao diện bạn nên giữ:
    predict_logits(model, loader, device, view=None) -> (filenames, y_true, logits[N, 9])
    aggregate_views(list_of_logits, space)           -> probs[N, 9]
    fit_temperature(val_logits, val_labels)          -> float T
    apply_temperature(logits, T)                     -> probs
    ensemble_probs(list_of_probs)                    -> probs
    fuse_conv_bn(model)                              -> model (BN đã gộp vào conv)
"""
from __future__ import annotations
import copy
import numpy as np
import torch
from torch.nn import functional as F


def predict_logits(model, loader, device, view=None):
    """Chạy model trên loader và gom logit theo đúng thứ tự file.

    `view` là hàm biến đổi batch ảnh trước khi đưa vào model (ví dụ lật ngang), hoặc None.
    TODO: model.eval(), torch.inference_mode(), (tuỳ chọn) autocast. Trả về numpy.
    """
    model.eval()
    names, labels, outputs = [], [], []
    with torch.inference_mode():
        for x, y, filename in loader:
            x = x.to(device, non_blocking=True)
            x = view(x) if view is not None else x
            logits = model(x)
            names.extend(filename)
            labels.append(y.numpy())
            outputs.append(logits.float().cpu().numpy())
    return names, np.concatenate(labels), np.concatenate(outputs)


def view_identity(x):
    return x


def view_hflip(x):
    """Lật ngang batch (N, C, H, W). TODO: dùng torch.flip trên chiều rộng (slide trang 75)."""
    return torch.flip(x, dims=(-1,))


def views_multicrop(x, crop: int):
    """5 crop (4 góc + giữa) kích thước `crop`, và tuỳ chọn thêm bản lật. Trả về list các batch. TODO."""
    h, w = x.shape[-2:]
    if crop > min(h, w) or crop <= 0:
        raise ValueError("crop không hợp lệ")
    positions = [(0, 0), (0, w-crop), (h-crop, 0), (h-crop, w-crop),
                 ((h-crop)//2, (w-crop)//2)]
    return [x[..., y:y+crop, x0:x0+crop] for y, x0 in positions]


def views_multiscale(x, sizes):
    """Resize batch về từng kích thước trong `sizes`, trả về list các batch. TODO.

    Lưu ý: model phải chấp nhận ảnh khác kích thước lúc train (CNN có global pooling thì được;
    ViT/Swin cần xử lý riêng vị trí/cửa sổ). Ghi rõ giới hạn bạn gặp.
    """
    return [F.interpolate(x, size=(int(size), int(size)), mode="bilinear", align_corners=False)
            for size in sizes]


def aggregate_views(logits_per_view, space: str = "prob"):
    """Gộp K lượt chạy của TTA thành một dự đoán (slide trang 62).

      - space="prob":  trung bình softmax của từng view
      - space="logit": trung bình logit rồi softmax
    Slide chưa kết luận cách nào luôn tốt hơn: chọn một và ghi rõ, hoặc so sánh cả hai (I03).
    TODO: trả về xác suất (N, 9) đã chuẩn hoá.
    """
    stack = np.stack([np.asarray(z, dtype=np.float64) for z in logits_per_view])
    if stack.ndim != 3 or len(stack) == 0:
        raise ValueError("Cần danh sách logits cùng dạng (N,K)")
    if space == "logit":
        z = stack.mean(axis=0)
        z -= z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)
    if space == "prob":
        z = stack - stack.max(axis=2, keepdims=True)
        e = np.exp(z)
        return (e / e.sum(axis=2, keepdims=True)).mean(axis=0)
    raise ValueError("space phải là prob hoặc logit")


def ensemble_probs(list_of_probs):
    """Trung bình xác suất của nhiều mô hình (khác backbone hoặc khác seed). TODO.

    Chi phí suy luận = số mô hình. Chỉ ghép các mô hình trên CÙNG tập ảnh và cùng thứ tự file.
    """
    stack = np.stack([np.asarray(p, dtype=np.float64) for p in list_of_probs])
    if stack.ndim != 3 or len(stack) == 0:
        raise ValueError("Cần danh sách xác suất cùng dạng (N,K)")
    if not np.allclose(stack.sum(axis=2), 1, atol=1e-5):
        raise ValueError("Xác suất chưa chuẩn hoá")
    return stack.mean(axis=0)


def fit_temperature(val_logits, val_labels) -> float:
    """Tìm nhiệt độ T > 0 cực tiểu NLL trên VAL: p = softmax(logit / T)  (slide trang 69).

    TODO: tối ưu hoá một tham số (LBFGS trên log T, hoặc tìm lưới thô rồi tinh).
    Accuracy không đổi vì thứ tự lớp không đổi. KHÔNG khớp T trên test.
    """
    z = torch.as_tensor(val_logits, dtype=torch.float64)
    y = torch.as_tensor(val_labels, dtype=torch.long)
    if z.ndim != 2 or len(z) != len(y) or len(y) == 0:
        raise ValueError("val_logits và val_labels không khớp")
    log_t = torch.zeros((), dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)
    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(z / log_t.exp(), y)
        loss.backward()
        return loss
    opt.step(closure)
    return float(log_t.detach().exp().item())


def apply_temperature(logits, T: float):
    """Trả về softmax(logits / T). TODO."""
    if T <= 0:
        raise ValueError("T phải dương")
    z = np.asarray(logits, dtype=np.float64) / T
    z -= z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def fuse_conv_bn(model):
    """Gộp BatchNorm vào tích chập liền trước, chính xác lúc suy luận (slide trang 71, 75):

        w' = gamma * w / sqrt(var + eps)        b' = beta + gamma * (b - mean) / sqrt(var + eps)

    TODO:
      - model.eval() trước
      - với từng cặp (Conv2d, BatchNorm2d) liền kề: tạo conv mới (có bias) và thay BN bằng Identity
      - kiểm tra: đầu ra trước/sau gộp lệch nhau cỡ 1e-5 trở xuống (in ra sai số lớn nhất)
    Với kiến trúc không có BN (ViT, Swin, ConvNeXt dùng LayerNorm), mục này không áp dụng; ghi rõ.
    """
    from torch.nn.utils.fusion import fuse_conv_bn_eval
    fused = copy.deepcopy(model).eval()
    def walk(parent):
        children = list(parent.named_children())
        for i in range(len(children) - 1):
            conv_name, conv = children[i]
            bn_name, bn = children[i + 1]
            if isinstance(conv, torch.nn.Conv2d) and isinstance(bn, torch.nn.BatchNorm2d):
                setattr(parent, conv_name, fuse_conv_bn_eval(conv, bn))
                setattr(parent, bn_name, torch.nn.Identity())
        for _, child in parent.named_children():
            walk(child)
    walk(fused)
    return fused
