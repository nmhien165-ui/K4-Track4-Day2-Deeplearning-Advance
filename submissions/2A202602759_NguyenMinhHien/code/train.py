"""train.py - vòng huấn luyện cho mọi thí nghiệm (B, T, F).

Đã hoàn thiện từ bộ khung (cấu hình và quy ước đặt tên file). Dùng MỘT hàm `run(cfg)` cho mọi cấu hình
(RUBRIC mục H): đổi thí nghiệm chỉ bằng cách đổi `Config`.

Chạy một thí nghiệm từ dòng lệnh:
    python train.py --set exp_id=B01 backbone=resnet50 seed=0
Chỉ số dùng để chọn checkpoint (macro-F1 val) phải tính bằng eval.compute_metrics của repo gốc,
để cùng định nghĩa với lúc chấm:
    sys.path.insert(0, "<thư mục chứa eval.py>");  from eval import compute_metrics
"""
from __future__ import annotations

from dataclasses import dataclass
from dataclasses import asdict, fields
from pathlib import Path
import argparse
import json
import math
import random
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from eval import compute_metrics, save_predictions
import dataset
import losses
import model as model_tools
import inference

# Ghi file dự đoán đúng định dạng bằng hàm có sẵn trong eval.py (repo gốc):
#     from eval import save_predictions, compute_metrics
# Log theo epoch (history.csv) và config.json bạn tự ghi bằng pandas/json.


@dataclass
class Config:
    # --- định danh ---
    exp_id: str = "T00"
    seed: int = 0
    fold: int = 0
    # --- mô hình ---
    backbone: str = "resnet50"
    init: str = "finetune"            # scratch | frozen | finetune
    drop_rate: float = 0.0
    # --- dữ liệu / augmentation ---
    img_size: int = 224
    aug: str = "basic"                # basic | color | trivial | randaug ...
    sampler: str | None = None        # None | balanced
    mix: str | None = None            # None | mixup | cutmix
    mix_alpha: float = 1.0
    # --- loss ---
    loss: str = "ce"                  # ce | ls | focal | ce_weighted
    label_smoothing: float = 0.0
    focal_gamma: float = 2.0
    class_weight_beta: float | None = None
    # --- tối ưu (công thức nền, GUIDE.md mục 1.4) ---
    epochs: int = 12
    batch_size: int = 64
    lr_backbone: float = 1e-4
    lr_head: float = 1e-3
    weight_decay: float = 0.05
    warmup_epochs: float = 1.0
    ema_decay: float | None = None
    amp: bool = True
    num_workers: int = 2
    # --- đường dẫn ---
    images_dir: str = "data/images"
    labels_dir: str = "data/labels"
    out_dir: str = "runs"             # config.json, history.csv, checkpoint, logit của từng lần chạy
    pred_dir: str = "predictions"     # file dự đoán đúng định dạng eval.py (nộp cùng bài)
    # --- chỉ bật ở Bước 4 (chung kết): ghi predictions trên TEST. Mặc định TẮT (quy tắc S4). ---
    save_test_predictions: bool = False
    inference_method: str = "i00"  # i00 | hflip_prob | hflip_logit | center_crop | temperature


def run_dir(cfg: Config) -> Path:
    """Thư mục kết quả của một lần chạy: <out_dir>/<exp_id>/seed<k>/ ."""
    return Path(cfg.out_dir) / cfg.exp_id / f"seed{cfg.seed}"


def pred_path(cfg: Config, split: str) -> Path:
    """Đường dẫn chuẩn của file dự đoán: <pred_dir>/<exp_id>_seed<k>_<split>.csv (split = val | test)."""
    return Path(cfg.pred_dir) / f"{cfg.exp_id}_seed{cfg.seed}_{split}.csv"


def set_seed(seed: int) -> None:
    """Cố định mọi nguồn ngẫu nhiên.

    TODO: random, numpy, torch (CPU và CUDA); cân nhắc cudnn.deterministic/benchmark và
    seed cho worker của DataLoader. Ghi lại trong báo cáo mức độ tái lập bạn đạt được.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_optimizer(model, cfg: Config):
    """AdamW với 3 nhóm tham số (xem model.param_groups). TODO."""
    return torch.optim.AdamW(model_tools.param_groups(model, cfg.lr_backbone,
                                                       cfg.lr_head, cfg.weight_decay))


def build_scheduler(optimizer, cfg: Config, steps_per_epoch: int):
    """Warmup tuyến tính rồi cosine về ~0 (slide trang 55). TODO.

    Cập nhật theo bước (iteration) hoặc theo epoch đều được; ghi rõ bạn chọn gì.
    Gợi ý kiểm tra: vẽ đường LR theo bước để thấy đúng hình warmup + cosine.
    """
    total = max(1, cfg.epochs * steps_per_epoch)
    warmup = int(cfg.warmup_epochs * steps_per_epoch)
    base = [group["lr"] for group in optimizer.param_groups]
    def factor(step):
        if warmup and step < warmup:
            return (step + 1) / warmup
        progress = min(1.0, (step - warmup) / max(1, total - warmup))
        return max(1e-6, 0.5 * (1 + math.cos(math.pi * progress)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, factor)
    scheduler.base_lrs = base
    return scheduler


class EMA:
    """Trung bình động trọng số: W_ema <- d * W_ema + (1 - d) * W  (slide trang 56).

    TODO:
      - __init__(model, decay): sao chép trọng số
      - update(model): sau mỗi bước tối ưu
      - copy_to(model) hoặc dùng bản sao riêng để đánh giá bằng trọng số EMA
      - lưu ý BatchNorm: buffer (running_mean/var) cũng phải được xử lý hợp lý
    """

    def __init__(self, model, decay: float):
        import copy
        if not 0 < decay < 1:
            raise ValueError("EMA decay phải thuộc (0,1)")
        self.decay = decay
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)

    def update(self, model) -> None:
        with torch.no_grad():
            for ema_p, p in zip(self.module.parameters(), model.parameters()):
                ema_p.lerp_(p.detach(), 1 - self.decay)
            for ema_b, b in zip(self.module.buffers(), model.buffers()):
                ema_b.copy_(b)

    def copy_to(self, model) -> None:
        model.load_state_dict(self.module.state_dict())


def train_one_epoch(model, loader, criterion, optimizer, scheduler, scaler, cfg: Config,
                    device, ema: EMA | None = None) -> dict:
    """Một epoch huấn luyện. Trả về dict, ví dụ {"train_loss": ..., "lr": ...}.

    TODO:
      - model.train() (nếu init == "frozen": giữ phần backbone ở eval, xem model.freeze_backbone)
      - nếu cfg.mix: mix_batch rồi mixed_loss (losses.py)
      - AMP (autocast + GradScaler), clip gradient nếu cần, optimizer.step(), scheduler.step()
      - nếu có EMA: ema.update(model)
    """
    model.train()
    if cfg.init == "frozen":
        for module in model.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                module.eval()
    total_loss, total_n = 0.0, 0
    start = time.perf_counter()
    for x, y, _ in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        if cfg.mix:
            x, targets = losses.mix_batch(x, y, cfg.mix_alpha, cfg.mix)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=cfg.amp and device.type == "cuda"):
            logits = model(x)
            loss = losses.mixed_loss(criterion, logits, targets) if cfg.mix else criterion(logits, y)
        if scaler is not None and scaler.is_enabled():
            previous_scale = scaler.get_scale()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer_stepped = scaler.get_scale() >= previous_scale
        else:
            loss.backward()
            optimizer.step()
            optimizer_stepped = True
        if optimizer_stepped:
            scheduler.step()
            if ema is not None:
                ema.update(model)
        total_loss += float(loss.detach()) * len(y)
        total_n += len(y)
    return {"train_loss": total_loss / total_n, "lr": optimizer.param_groups[0]["lr"],
            "train_seconds": time.perf_counter() - start}


def evaluate(model, loader, criterion, device):
    """Chạy model trên một loader ở chế độ eval, KHÔNG tính gradient.

    Trả về (filenames: list[str], y_true: ndarray[N], logits: ndarray[N, 9], loss: float).
    Giữ đúng thứ tự của loader để ghép logit với tên file.

    TODO: model.eval(), torch.inference_mode(), gom kết quả. Softmax khi cần xác suất.
    """
    model.eval()
    names, labels, outputs = [], [], []
    total_loss, total_n = 0.0, 0
    with torch.inference_mode():
        for x, y, filename in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            logits = model(x)
            loss = criterion(logits, y)
            names.extend(filename)
            labels.append(y.cpu().numpy())
            outputs.append(logits.float().cpu().numpy())
            total_loss += float(loss) * len(y)
            total_n += len(y)
    return names, np.concatenate(labels), np.concatenate(outputs), total_loss / total_n


def plot_curves(history: list[dict], path: str | Path, title: str) -> None:
    """Vẽ đường cong training của một thí nghiệm -> curves/<exp_id>_<mota>.png (GUIDE.md mục 6.2).

    TODO: tối thiểu loss train/val và macro-F1 val theo epoch; có tiêu đề, nhãn trục, chú thích;
    khuyến khích thêm LR theo bước. Lưu bằng matplotlib với dpi đủ nét để đọc số.
    """
    import matplotlib.pyplot as plt
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(history)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(frame["epoch"], frame["train_loss"], label="train loss")
    axes[0].plot(frame["epoch"], frame["val_loss"], label="val loss")
    axes[0].set(xlabel="Epoch", ylabel="Loss")
    axes[0].legend()
    axes[1].plot(frame["epoch"], frame["macro_f1_val"], label="macro-F1 val")
    axes[1].set(xlabel="Epoch", ylabel="Macro-F1", ylim=(0, 1))
    axes[1].legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def predict_with_method(net, loader, device, method: str, temperature: float | None = None):
    """Dự đoán một split theo phương pháp đã chốt trên val; trả tên, nhãn, xác suất."""
    if method == "resolution256":
        names, y, logits = inference.predict_logits(net, loader, device)
        return names, y, logits, inference.apply_temperature(logits, 1.0)
    names, y, logits = inference.predict_logits(net, loader, device)
    if method == "i00":
        probs = inference.apply_temperature(logits, 1.0)
    elif method in ("hflip_prob", "hflip_logit"):
        flipped_names, flipped_y, flipped = inference.predict_logits(
            net, loader, device, inference.view_hflip)
        if names != flipped_names or not np.array_equal(y, flipped_y):
            raise ValueError("TTA không khớp thứ tự Filename")
        probs = inference.aggregate_views([logits, flipped],
                                          "prob" if method == "hflip_prob" else "logit")
    elif method == "center_crop":
        def crop_view(x):
            cropped = inference.views_multicrop(x, int(x.shape[-1] * 0.9))[-1]
            return torch.nn.functional.interpolate(cropped, size=x.shape[-2:],
                                                    mode="bilinear", align_corners=False)
        cropped_names, cropped_y, cropped_logits = inference.predict_logits(
            net, loader, device, crop_view)
        if names != cropped_names or not np.array_equal(y, cropped_y):
            raise ValueError("Crop TTA không khớp thứ tự Filename")
        probs = inference.aggregate_views([logits, cropped_logits], "prob")
    elif method == "temperature":
        if temperature is None:
            raise ValueError("temperature phải được khớp trên val trước")
        probs = inference.apply_temperature(logits, temperature)
    else:
        raise ValueError(f"inference_method không hỗ trợ: {method}")
    return names, y, logits, probs


def run(cfg: Config) -> dict:
    """Huấn luyện một cấu hình và lưu mọi thứ cần thiết. Trả về dict kết quả tóm tắt.

    TODO theo thứ tự:
      1. set_seed; tạo thư mục run_dir(cfg); ghi config.json (dataclasses.asdict(cfg))
      2. dataset.load_split + dataset.check_split (dừng nếu vi phạm S1-S6)
      3. dựng train/val loader (test loader chỉ tạo khi cfg.save_test_predictions)
      4. model.build_model, criterion (losses.build_criterion), optimizer, scheduler, scaler, EMA
      5. với mỗi epoch: train_one_epoch -> evaluate(val) -> ghi history (loss, macro-F1 val, lr...)
         và lưu checkpoint tốt nhất theo MACRO-F1 VAL (hòa thì lấy epoch sớm hơn)
      6. cuối: nạp checkpoint tốt nhất, lưu val logits và eval.save_predictions(pred_path(cfg, "val"), ...)
      7. NẾU cfg.save_test_predictions (chỉ ở Bước 4): đánh giá test đúng MỘT lần,
         lưu logits và eval.save_predictions(pred_path(cfg, "test"), ...)
      8. ghi history.csv, plot_curves(...), trả về dict tóm tắt
         (best_epoch, macro-F1 val, thời gian train mỗi epoch, số tham số, GMAC)
    Quy tắc: KHÔNG dùng test để chọn checkpoint hay bất kỳ quyết định nào (README.md, S4).
    """
    if cfg.epochs < 1 or cfg.fold != 0:
        raise ValueError("Bài nộp chính dùng fold 0 và epochs >= 1")
    set_seed(cfg.seed)
    output = run_dir(cfg)
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")
    train_df, val_df, test_df = dataset.load_split(cfg.labels_dir, cfg.fold)
    split_stats = dataset.check_split(train_df, val_df, test_df, cfg.images_dir)
    (output / "split_check.json").write_text(json.dumps(split_stats, indent=2), encoding="utf-8")
    train_loader = dataset.make_loader(train_df, cfg.images_dir,
                                       dataset.build_transforms(True, cfg.img_size, cfg.aug),
                                       cfg.batch_size, True, cfg.sampler, cfg.num_workers)
    eval_transform = dataset.build_transforms(False, cfg.img_size, cfg.aug)
    val_loader = dataset.make_loader(val_df, cfg.images_dir, eval_transform,
                                     cfg.batch_size, False, num_workers=cfg.num_workers)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = model_tools.build_model(cfg.backbone, pretrained=cfg.init != "scratch",
                                  drop_rate=cfg.drop_rate, init=cfg.init).to(device)
    counts = train_df["Label"].value_counts().reindex(range(dataset.NUM_CLASSES), fill_value=0).to_numpy()
    weight = losses.class_weights(counts, cfg.class_weight_beta or 0.0).to(device) if cfg.loss == "ce_weighted" else None
    criterion = losses.build_criterion(cfg.loss, smoothing=cfg.label_smoothing,
                                       gamma=cfg.focal_gamma, weight=weight).to(device)
    optimizer = build_optimizer(net, cfg)
    scheduler = build_scheduler(optimizer, cfg, len(train_loader))
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp and device.type == "cuda")
    ema = EMA(net, cfg.ema_decay) if cfg.ema_decay is not None else None
    history = []
    best_f1, best_epoch = -1.0, -1
    for epoch in range(cfg.epochs):
        train_stats = train_one_epoch(net, train_loader, criterion, optimizer,
                                      scheduler, scaler, cfg, device, ema)
        evaluated = ema.module if ema is not None else net
        _, y_val, logits_val, val_loss = evaluate(evaluated, val_loader, criterion, device)
        probs_val = torch.softmax(torch.from_numpy(logits_val), dim=1).numpy()
        metrics = compute_metrics(y_val, probs_val.argmax(axis=1), probs_val)
        row = {"epoch": epoch + 1, **train_stats, "val_loss": val_loss,
               "macro_f1_val": metrics["macro_f1"], "top1_val": metrics["top1"]}
        history.append(row)
        pd.DataFrame(history).to_csv(output / "history.csv", index=False)
        print(f"{cfg.exp_id} seed{cfg.seed} epoch {epoch + 1}/{cfg.epochs}: "
              f"train_loss={row['train_loss']:.4f} val_loss={val_loss:.4f} "
              f"macro_f1_val={metrics['macro_f1']:.4f} "
              f"top1_val={metrics['top1']:.4f} train_s={row['train_seconds']:.1f}", flush=True)
        checkpoint = {"epoch": epoch + 1, "model": evaluated.state_dict(),
                      "macro_f1_val": metrics["macro_f1"], "config": asdict(cfg)}
        torch.save(checkpoint, output / "last.pt")
        if metrics["macro_f1"] > best_f1:
            best_f1, best_epoch = metrics["macro_f1"], epoch + 1
            torch.save(checkpoint, output / "best.pt")
    checkpoint = torch.load(output / "best.pt", map_location=device, weights_only=False)
    net.load_state_dict(checkpoint["model"])
    names, y_val, logits_val, _ = evaluate(net, val_loader, criterion, device)
    np.save(output / "val_logits.npy", logits_val)
    temperature = inference.fit_temperature(logits_val, y_val) if cfg.inference_method == "temperature" else None
    if cfg.inference_method == "temperature":
        (output / "temperature.json").write_text(json.dumps({"T": temperature, "fit_split": "val"}), encoding="utf-8")
    if cfg.inference_method == "i00":
        probs_val = inference.apply_temperature(logits_val, 1.0)
    else:
        inference_val_loader = val_loader
        if cfg.inference_method == "resolution256":
            inference_val_loader = dataset.make_loader(
                val_df, cfg.images_dir, dataset.build_transforms(False, 256),
                cfg.batch_size, False, num_workers=cfg.num_workers)
        names, y_val, _, probs_val = predict_with_method(net, inference_val_loader, device,
                                                         cfg.inference_method, temperature)
    save_predictions(pred_path(cfg, "val"), names, y_val, probs_val)
    if cfg.save_test_predictions:
        test_transform = (dataset.build_transforms(False, 256)
                          if cfg.inference_method == "resolution256" else eval_transform)
        test_loader = dataset.make_loader(test_df, cfg.images_dir, test_transform,
                                          cfg.batch_size, False, num_workers=cfg.num_workers)
        names, y_test, logits_test, probs_test = predict_with_method(
            net, test_loader, device, cfg.inference_method, temperature)
        np.save(output / "test_logits.npy", logits_test)
        save_predictions(pred_path(cfg, "test"), names, y_test, probs_test)
    curve = Path(__file__).resolve().parent.parent / "curves" / f"{cfg.exp_id}_{cfg.backbone}_seed{cfg.seed}.png"
    plot_curves(history, curve, f"{cfg.exp_id} | {cfg.backbone} | seed {cfg.seed}")
    result = {"exp_id": cfg.exp_id, "seed": cfg.seed, "backbone": cfg.backbone,
              "weight_tag": net.weight_tag, "best_epoch": best_epoch,
              "macro_f1_val": compute_metrics(y_val, probs_val.argmax(1), probs_val)["macro_f1"],
              "checkpoint_macro_f1_val_i00": best_f1,
              "top1_val": compute_metrics(y_val, probs_val.argmax(1), probs_val)["top1"],
              "inference_method": cfg.inference_method, "temperature": temperature,
              "train_seconds_per_epoch": float(np.mean([h["train_seconds"] for h in history])),
              "params_m": model_tools.count_params(net),
              "gmac": model_tools.count_gmacs(net, cfg.img_size),
              "torch": torch.__version__, "device": str(device), "curve": str(curve)}
    (output / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def parse_overrides(pairs: list[str]) -> dict:
    """Biến ['seed=1', 'loss=focal', 'ema_decay=none'] thành dict, ép kiểu theo field của Config.

    TODO: tách key/value, báo lỗi rõ nếu key không có trong Config, ép int/float/bool/None theo kiểu field.
    """
    defaults = asdict(Config())
    result = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"Cần KEY=VALUE: {pair}")
        key, value = pair.split("=", 1)
        if key not in defaults:
            raise ValueError(f"Config không có trường: {key}")
        original = defaults[key]
        if value.lower() in ("none", "null"):
            if original is not None and key not in ("ema_decay", "sampler", "mix", "class_weight_beta"):
                raise ValueError(f"{key} không nhận None")
            result[key] = None
        elif isinstance(original, bool):
            if value.lower() not in ("true", "false", "1", "0"):
                raise ValueError(f"{key} cần true/false")
            result[key] = value.lower() in ("true", "1")
        elif isinstance(original, int):
            result[key] = int(value)
        elif isinstance(original, float) or key in ("ema_decay", "class_weight_beta"):
            result[key] = float(value)
        else:
            result[key] = value
    return result


def main() -> None:
    """Điểm vào dòng lệnh: `python train.py --set exp_id=B01 backbone=resnet50 seed=0`.

    TODO: argparse nhận `--set KEY=VALUE ...`, dựng Config qua parse_overrides, gọi run(cfg), in kết quả.
    """
    parser = argparse.ArgumentParser(description="Huấn luyện DeepWeeds fold 0")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()
    result = run(Config(**parse_overrides(args.set)))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
