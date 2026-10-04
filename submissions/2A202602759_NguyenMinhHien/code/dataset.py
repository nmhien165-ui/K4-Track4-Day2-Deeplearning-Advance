"""dataset.py - đọc DeepWeeds, kiểm tra chia dữ liệu, transform, DataLoader.

Đã hoàn thiện từ bộ khung pseudo-code của bài lab.
Quy tắc chia dữ liệu bắt buộc (S1-S6) nằm ở README.md, mục 2.1. Đọc trước khi viết.

Giao diện bạn phải giữ (để notebook, train.py và eval.py ghép được với nhau):
    load_split(labels_dir, fold=0)            -> (train_df, val_df, test_df)
    check_split(train_df, val_df, test_df, images_dir) -> dict  (số liệu để ghi báo cáo)
    build_transforms(train, img_size, aug)    -> torchvision transform
    DeepWeedsDataset[i]                       -> (image_tensor, label:int, filename:str)
    make_loader(df, images_dir, transform, batch_size, train, sampler, num_workers)
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from torch.utils.data import Dataset

NUM_CLASSES = 9
# Thứ tự lớp theo cột `Label` của labels.csv (0 = Chinee Apple ... 7 = Snake Weed, 8 = Negatives).
CLASS_NAMES = [
    "Chinee Apple", "Lantana", "Parkinsonia", "Parthenium", "Prickly Acacia",
    "Rubber Vine", "Siam Weed", "Snake Weed", "Negatives",
]
IMAGENET_MEAN = (0.485, 0.456, 0.406)  # đổi nếu trọng số timm bạn dùng yêu cầu mean/std khác
IMAGENET_STD = (0.229, 0.224, 0.225)


def seed_worker(worker_id):
    import random
    import numpy as np
    import torch
    seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(seed)
    random.seed(seed)


def load_split(labels_dir: str | Path, fold: int = 0):
    """Đọc train_subset{fold}.csv, val_subset{fold}.csv, test_subset{fold}.csv (S1).

    Các file subset gốc có `Filename,Label`; `Species` nằm trong labels.csv.
    Ghép Species theo Label của subset; Label trong subset là nhãn dùng để train/eval.
    Một ảnh train trong bản gốc có Label khác labels.csv, nên không ghi đè Label fold.
    KHÔNG sửa, lọc hay chia lại dữ liệu.

    TODO:
      - đọc ba file CSV bằng pandas
      - trả về (train_df, val_df, test_df)
    """
    if not 0 <= fold <= 4:
        raise ValueError("fold phải nằm trong 0..4")
    root = Path(labels_dir)
    labels = pd.read_csv(root / "labels.csv")
    if not {"Filename", "Label", "Species"} <= set(labels.columns):
        raise ValueError("labels.csv: thiếu cột Filename, Label hoặc Species")
    if labels["Filename"].isna().any() or labels["Filename"].duplicated().any():
        raise ValueError("labels.csv: Filename rỗng hoặc trùng")
    lookup = labels.set_index("Filename")
    species_by_label = labels.groupby("Label")["Species"].agg(lambda values: values.mode().iloc[0]).to_dict()
    frames = []
    for part in ("train", "val", "test"):
        frame = pd.read_csv(root / f"{part}_subset{fold}.csv")
        if not {"Filename", "Label"} <= set(frame.columns):
            raise ValueError(f"{part}: thiếu cột Filename hoặc Label")
        canonical = lookup.reindex(frame["Filename"])
        if canonical["Label"].isna().any():
            raise ValueError(f"{part}: Filename thiếu trong labels.csv")
        if not frame["Label"].isin(species_by_label).all():
            raise ValueError(f"{part}: Label không có trong labels.csv")
        frame = frame.copy()
        frame["Species"] = frame["Label"].map(species_by_label)
        frames.append(frame)
    return tuple(frames)


def check_split(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
                images_dir: str | Path) -> dict:
    """Kiểm tra bắt buộc trước khi train (README.md, mục 2.1). In ra và trả về dict số liệu.

    TODO kiểm tra, mỗi ý lỗi thì `assert` / raise để dừng ngay:
      1. số ảnh mỗi tập và số ảnh mỗi lớp trong từng tập (kỳ vọng xấp xỉ 60/20/20)
      2. giao của từng cặp tập theo Filename phải RỖNG (train∩val, train∩test, val∩test)
      3. hợp ba tập phải bằng đúng 17.509 ảnh
      4. mọi Filename đều tồn tại trong `images_dir`
    Trả về dict, ví dụ {"n": {...}, "per_class": {...}, "overlap": {...}} để dán vào báo cáo.
    """
    frames = {"train": train_df, "val": val_df, "test": test_df}
    names = {}
    per_class = {}
    for part, frame in frames.items():
        if frame["Filename"].isna().any() or frame["Filename"].duplicated().any():
            raise ValueError(f"{part}: Filename rỗng hoặc trùng")
        labels = pd.to_numeric(frame["Label"], errors="raise")
        if labels.isna().any() or not labels.isin(range(NUM_CLASSES)).all():
            raise ValueError(f"{part}: Label ngoài 0..8")
        names[part] = set(frame["Filename"].astype(str))
        per_class[part] = {int(k): int(v) for k, v in labels.value_counts().sort_index().items()}
    overlap = {"train_val": len(names["train"] & names["val"]),
               "train_test": len(names["train"] & names["test"]),
               "val_test": len(names["val"] & names["test"])}
    if any(overlap.values()):
        raise ValueError(f"Các split bị trùng Filename: {overlap}")
    union = set.union(*names.values())
    if len(union) != 17509:
        raise ValueError(f"Hợp ba split có {len(union)} ảnh, cần 17509")
    root = Path(images_dir).resolve()
    missing = []
    for filename in union:
        path = (root / filename).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            missing.append(filename)
    if missing:
        raise FileNotFoundError(f"Thiếu {len(missing)} ảnh, ví dụ: {missing[:5]}")
    n = {part: len(frame) for part, frame in frames.items()}
    ratios = {part: count / len(union) for part, count in n.items()}
    if any(abs(ratios[part] - expected) > 0.01 for part, expected in
           (("train", 0.6), ("val", 0.2), ("test", 0.2))):
        raise ValueError(f"Tỉ lệ split lệch quá 1 điểm phần trăm: {ratios}")
    result = {"n": n, "per_class": per_class, "overlap": overlap,
              "union": len(union), "ratio": ratios, "missing": 0}
    print(result)
    return result


def build_transforms(train: bool, img_size: int = 224, aug: str = "basic"):
    """Tạo transform. `aug` chọn mức augmentation; bạn tự định nghĩa các giá trị.

    Gợi ý các giá trị `aug` (trục B của GUIDE.md mục 3): "basic", "color", "trivial", "randaug".
    Mixup/CutMix trộn theo batch nên nằm ở losses.py, không ở đây.

    Train (basic): RandomResizedCrop(img_size) + lật ngang + ToTensor + Normalize.
    Val/test: ảnh gốc 256x256 -> CenterCrop(img_size) (hoặc giữ nguyên 256; ghi rõ bạn chọn gì)
              + ToTensor + Normalize. KHÔNG augmentation ngẫu nhiên khi đánh giá.

    TODO: dùng torchvision.transforms (hoặc v2). Lưu ý: lật dọc có hợp lệ với ảnh cỏ dại không?
    """
    from torchvision import transforms as T
    if img_size <= 0:
        raise ValueError("img_size phải dương")
    if not train:
        return T.Compose([T.Resize(256), T.CenterCrop(img_size), T.ToTensor(),
                          T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    ops = [T.RandomResizedCrop(img_size), T.RandomHorizontalFlip()]
    if aug == "color":
        ops.append(T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05))
    elif aug == "trivial":
        ops.append(T.TrivialAugmentWide())
    elif aug == "randaug":
        ops.append(T.RandAugment())
    elif aug != "basic":
        raise ValueError(f"aug không hỗ trợ: {aug}")
    return T.Compose(ops + [T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])


class DeepWeedsDataset(Dataset):
    """Dataset đọc ảnh từ `images_dir` theo DataFrame (Filename, Label).

    __getitem__(i) phải trả về (ảnh đã transform, nhãn int, tên file str).
    Tên file cần có để ghi `predictions/*.csv` đúng định dạng của eval.py.

    TODO:
      - __init__(self, df, images_dir, transform): giữ df, mở ảnh bằng PIL, chuyển sang RGB
      - __len__
      - __getitem__ -> (tensor, int(label), filename)
      - (tuỳ chọn) nạp trước ảnh vào RAM nếu bị nghẽn đọc đĩa trên Colab
    """

    def __init__(self, df: pd.DataFrame, images_dir: str | Path, transform=None):
        self.df = df.reset_index(drop=True).copy()
        self.images_dir = Path(images_dir)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int):
        from PIL import Image
        row = self.df.iloc[i]
        filename = str(row["Filename"])
        with Image.open(self.images_dir / filename) as image:
            image = image.convert("RGB")
            tensor = self.transform(image) if self.transform else image.copy()
        return tensor, int(row["Label"]), filename


def make_loader(df: pd.DataFrame, images_dir: str | Path, transform, batch_size: int,
                train: bool, sampler: str | None = None, num_workers: int = 2):
    """Tạo DataLoader.

    TODO:
      - train=True: shuffle (hoặc dùng sampler); train=False: không shuffle, giữ thứ tự df
        (thứ tự phải ổn định để ghép logit với Filename)
      - sampler=None | "balanced": "balanced" dùng WeightedRandomSampler với trọng số
        1/(số ảnh của lớp) (trục D của GUIDE.md mục 3)
      - drop_last=True khi train nếu batch cuối quá nhỏ làm BatchNorm không ổn định
      - pin_memory=True, num_workers hợp lý; seed cho worker (worker_init_fn) để tái lập
    """
    import torch
    from torch.utils.data import DataLoader, WeightedRandomSampler
    dataset = DeepWeedsDataset(df, images_dir, transform)
    weighted = None
    if sampler not in (None, "balanced"):
        raise ValueError(f"sampler không hỗ trợ: {sampler}")
    if train and sampler == "balanced":
        labels = df["Label"].astype(int)
        counts = labels.value_counts()
        weights = torch.as_tensor([1.0 / counts[int(y)] for y in labels], dtype=torch.double)
        weighted = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    return DataLoader(dataset, batch_size=batch_size, shuffle=train and weighted is None,
                      sampler=weighted, num_workers=num_workers, pin_memory=torch.cuda.is_available(),
                      drop_last=train, worker_init_fn=seed_worker)
