"""Спільний код: дані, модель, режими навчання, метрики."""
import csv
import hashlib
import sys
from pathlib import Path

import numpy as np
import torch
import torchvision
from PIL import Image
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from torchvision.models import ResNet18_Weights

ROOT = Path(__file__).resolve().parent
SPLIT = ROOT / "split.csv"
RUNS = ROOT / "runs"
RESULTS = ROOT / "results"
MODEL = ROOT / "lab3_model.pt"

CLASSES = ["buildings", "forest", "glacier", "mountain", "sea", "street"]
LABEL_ID = {c: i for i, c in enumerate(CLASSES)}
IMG_SIZE = 160
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
N_TRAIN = 200
EPOCHS = 8
BATCH, EVAL_BATCH = 32, 64
WEIGHT_DECAY = 1e-4

BODY = ("conv1", "bn1", "layer1", "layer2", "layer3", "layer4")
# Режим: чи беремо ваги ImageNet і групи оптимізатора (модулі, швидкість навчання)
REGIMES = {
    "scratch": {"pretrained": False, "groups": [(BODY + ("fc",), 3e-3)]},
    "frozen": {"pretrained": True, "groups": [(("fc",), 3e-3)]},
    "unfreeze": {"pretrained": True, "groups": [(("fc",), 3e-3), (("layer4",), 3e-4)]},
}
SEEDS = {"scratch": (1,), "frozen": (1, 2, 3), "unfreeze": (1, 2, 3)}
RUN_LIST = [(r, s) for r in REGIMES for s in SEEDS[r]]

_NORM = [T.ToTensor(), T.Normalize(MEAN, STD)]
TRAIN_TF = T.Compose(
    [T.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)), T.RandomHorizontalFlip(), *_NORM]
)
EVAL_TF = T.Compose([T.Resize((IMG_SIZE, IMG_SIZE)), *_NORM])


# Поділ

def image_path(data, row):
    folder = "seg_train" if row["part"] == "train" else "seg_test"
    return Path(data) / folder / folder / row["label"] / row["file"]


def make_split(data):
    """Будує split.csv за процедурою завдання і записує його."""
    def names(folder, label):
        files = sorted(p.name for p in (Path(data) / folder / folder / label).glob("*.jpg"))
        if not files:
            sys.exit(f"немає зображень у {folder}/{folder}/{label}")
        return files

    train, val, test = [], [], []
    rng = np.random.default_rng(0)
    for label in CLASSES:
        files = names("seg_train", label)
        idx = np.arange(len(files))
        rng.shuffle(idx)
        train += [(files[i], label, "train") for i in sorted(idx[:N_TRAIN])]
    rng = np.random.default_rng(0)
    for label in CLASSES:
        files = names("seg_test", label)
        idx = np.arange(len(files))
        rng.shuffle(idx)
        cut = len(files) // 2
        val += [(files[i], label, "validation") for i in sorted(idx[:cut])]
        test += [(files[i], label, "test") for i in sorted(idx[cut:])]

    with open(SPLIT, "w", newline="", encoding="utf-8") as f:
        # Однаковий символ кінця рядка на всіх ОС, щоб збігався хеш файлу
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["file", "label", "part"])
        writer.writerows(train + val + test)


def read_split():
    with open(SPLIT, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def part_rows(rows, part):
    return [r for r in rows if r["part"] == part]


def label_ids(rows):
    return np.array([LABEL_ID[r["label"]] for r in rows])


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# Дані

class Images(Dataset):
    def __init__(self, rows, data, transform):
        self.rows, self.data, self.transform = rows, data, transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        img = Image.open(image_path(self.data, row)).convert("RGB")
        return self.transform(img), LABEL_ID[row["label"]]


def train_loader(rows, data):
    return DataLoader(
        Images(rows, data, TRAIN_TF), batch_size=BATCH, shuffle=True, drop_last=False, num_workers=0
    )


def eval_loader(rows, data):
    return DataLoader(
        Images(rows, data, EVAL_TF), batch_size=EVAL_BATCH, shuffle=False, num_workers=0
    )


# Модель і режими

def trainable(regime):
    return {m for mods, _ in REGIMES[regime]["groups"] for m in mods}


def eval_modules(regime):
    """Модулі, що лишаються в eval() під час навчання."""
    return [m for m in BODY if m not in trainable(regime)]


def build_model(regime, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    weights = ResNet18_Weights.IMAGENET1K_V1 if REGIMES[regime]["pretrained"] else None
    model = torchvision.models.resnet18(weights=weights)
    model.fc = nn.Linear(512, len(CLASSES))
    for name, p in model.named_parameters():
        p.requires_grad = name.split(".")[0] in trainable(regime)
    return model


def make_optimizer(model, regime):
    groups = [
        {"params": [p for m in mods for p in getattr(model, m).parameters()], "lr": lr}
        for mods, lr in REGIMES[regime]["groups"]
    ]
    return torch.optim.AdamW(groups, weight_decay=WEIGHT_DECAY)


def set_train_mode(model, regime):
    model.train()
    for name in eval_modules(regime):
        getattr(model, name).eval()


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# Метрики

def predict_logits(model, loader, device):
    model.eval()
    out = []
    with torch.no_grad():
        for x, _ in loader:
            out.append(model(x.to(device)).cpu())
    return torch.cat(out).numpy()


def macro_f1(y_true, logits):
    return f1_score(y_true, logits.argmax(1), average="macro", labels=list(range(len(CLASSES))))


def select(f1):
    """Режим з найбільшим середнім F, у ньому запуск з найбільшим F.

    f1: словник (режим, зерно) -> F на validation після останньої епохи.
    """
    mean = {r: np.mean([f1[(r, s)] for s in SEEDS[r]]) for r in REGIMES}
    regime = max(REGIMES, key=lambda r: mean[r])
    seed = max(SEEDS[regime], key=lambda s: f1[(regime, s)])
    return regime, seed, mean


def md_table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + " --- |" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)
