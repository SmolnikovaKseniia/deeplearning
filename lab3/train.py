"""Навчання: поділ, перевірки до навчання, сім запусків, вибір і збереження моделі.

Запуск: uv run python train.py --data path/to/intel
"""
import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # без вікон, працює на будь-якій ОС
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from torch import nn

from common import (
    BATCH, CLASSES, EPOCHS, EVAL_TF, IMG_SIZE, MEAN, MODEL, REGIMES, RESULTS, RUN_LIST, RUNS,
    SPLIT, STD, TRAIN_TF, WEIGHT_DECAY, build_model, eval_loader, eval_modules, file_sha256,
    label_ids, macro_f1, make_optimizer, make_split, md_table, part_rows, pick_device,
    predict_logits, read_split, select, set_train_mode, train_loader, trainable,
)

# Контрольна таблиця поділу: клас -> частина -> (кількість, сума номерів у іменах файлів)
EXPECTED_SPLIT = {
    "buildings": {"train": (200, 1996761), "validation": (218, 4837323), "test": (219, 4837559)},
    "forest": {"train": (200, 2032625), "validation": (237, 5276962), "test": (237, 5291326)},
    "glacier": {"train": (200, 2183923), "validation": (276, 6135652), "test": (277, 6157250)},
    "mountain": {"train": (200, 1950345), "validation": (262, 5789535), "test": (263, 5849765)},
    "sea": {"train": (200, 1987613), "validation": (255, 5655068), "test": (255, 5668011)},
    "street": {"train": (200, 1841817), "validation": (250, 5541845), "test": (251, 5547533)},
}
PARTS = ("train", "validation", "test")
TOTAL_PARAMS = 11179590
TRAIN_PARAMS = {"scratch": 11179590, "frozen": 3078, "unfreeze": 8396806}

REPORT = []


def say(text=""):
    print(text)
    REPORT.append(text)


def yn(flag):
    return "так" if flag else "ні"


# Перевірки до навчання

def check_split(rows):
    table, ok = [], True
    totals = {p: [0, 0] for p in PARTS}
    for label in CLASSES:
        cells = []
        for part in PARTS:
            names = [r["file"] for r in rows if r["label"] == label and r["part"] == part]
            got = (len(names), sum(int(Path(n).stem) for n in names))
            good = got == EXPECTED_SPLIT[label][part]
            ok &= good
            totals[part][0] += got[0]
            totals[part][1] += got[1]
            cells.append(f"{got[0]} / {got[1]} ({yn(good)})")
        table.append([label, *cells])
    table.append(["Разом", *[f"{n} / {s}" for n, s in totals.values()]])
    files = [r["file"] for r in rows]
    unique = len(files) == len(set(files))
    say("## Поділ\n")
    say(md_table(["Клас", *PARTS], table))
    say(f"\nУсі клітинки збігаються: {yn(ok)}. Жоден файл не входить у дві частини: {yn(unique)}.")
    return ok and unique


def save_batch_figure(rows, data, path):
    torch.manual_seed(1)
    x, y = next(iter(train_loader(part_rows(rows, "train"), data)))
    mean = torch.tensor(MEAN).view(1, 3, 1, 1)
    std = torch.tensor(STD).view(1, 3, 1, 1)
    img = (x * std + mean).clamp(0, 1)  # обернене нормування
    fig, axes = plt.subplots(4, 4, figsize=(8, 8.5))
    for ax, im, label in zip(axes.flat, img[:16], y[:16]):
        ax.imshow(im.permute(1, 2, 0))
        ax.set_title(CLASSES[label], fontsize=9)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def initial_loss(regime, rows, data, device):
    model = build_model(regime, 1).to(device)
    x, y = next(iter(train_loader(part_rows(rows, "train"), data)))
    set_train_mode(model, regime)
    with torch.no_grad():
        return nn.CrossEntropyLoss()(model(x.to(device)), y.to(device)).item()


# Перевірка режимів

def check_regimes(device):
    table, ok = [], True
    for regime, cfg in REGIMES.items():
        model = build_model(regime, 1).to(device)
        opt = make_optimizer(model, regime)
        total = sum(p.numel() for p in model.parameters())
        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        in_opt = {id(p) for g in opt.param_groups for p in g["params"]}
        leak = any(id(p) in in_opt for p in model.parameters() if not p.requires_grad)
        groups = "; ".join(
            f"{', '.join(mods)}: {sum(p.numel() for p in g['params'])} при {g['lr']:g}"
            for (mods, _), g in zip(cfg["groups"], opt.param_groups)
        )
        good = total == TOTAL_PARAMS and n_train == TRAIN_PARAMS[regime] and not leak
        ok &= good
        table.append([regime, total, n_train, groups, yn(leak), yn(good)])
    say("\n## Режими\n")
    say(md_table(["Режим", "Усього", "requires_grad", "Групи оптимізатора", "Заморожений у оптимізаторі", "Збігається"], table))
    return ok


def plain_train(model, regime):
    """Навмисна помилка: усе в train(), включно з замороженим кістяком."""
    model.train()


def five_steps(regime, rows, data, device, mode):
    """Копія стану, перемикання режиму модулів, 5 кроків оптимізатора."""
    model = build_model(regime, 1).to(device)
    opt = make_optimizer(model, regime)
    loader = train_loader(part_rows(rows, "train"), data)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    mode(model, regime)
    loss_fn = nn.CrossEntropyLoss()
    for i, (x, y) in enumerate(loader):
        if i == 5:
            break
        opt.zero_grad()
        loss_fn(model(x.to(device)), y.to(device)).backward()
        opt.step()
    return model, before


def change_table(regime, model, before):
    table, ok = [], True
    for name, mod in model.named_children():
        params, bufs = list(mod.named_parameters()), list(mod.named_buffers())
        if not params and not bufs:
            continue
        p_changed = any(not torch.equal(before[f"{name}.{k}"], v) for k, v in params)
        s_changed = any(not torch.equal(before[f"{name}.{k}"], v) for k, v in bufs)
        p_expected = name in trainable(regime)
        s_expected = bool(bufs) and name not in eval_modules(regime)
        good = p_changed == p_expected and s_changed == s_expected
        ok &= good
        table.append([regime, name, yn(p_changed), yn(s_changed) if bufs else "немає BN", yn(good)])
    return table, ok


def bn_change(model, before, key):
    now = dict(model.named_buffers())[key]
    return (now - before[key]).abs().max().item()


def check_changes(rows, data, device):
    header = ["Режим", "Модуль", "Параметри змінилися", "Статистики BN змінилися", "Збігається з очікуваним"]
    table, ok = [], True
    for regime in ("frozen", "unfreeze"):
        model, before = five_steps(regime, rows, data, device, set_train_mode)
        part, good = change_table(regime, model, before)
        table += part
        ok &= good
    say("\n## Що змінюється під час навчання\n")
    say(md_table(header, table))

    # Дослід із навмисною помилкою
    model, before = five_steps("frozen", rows, data, device, plain_train)
    table, _ = change_table("frozen", model, before)
    say("\n## Дослід: frozen з model.train() замість set_train_mode\n")
    say(md_table(header, table))
    mean_change = bn_change(model, before, "layer1.0.bn1.running_mean")
    var_change = bn_change(model, before, "layer1.0.bn1.running_var")
    say(f"\nlayer1.0.bn1 після 5 кроків: max |зміна running_mean| = {mean_change:.6g}, "
        f"max |зміна running_var| = {var_change:.6g}.")
    return ok and mean_change > 0 and var_change > 0


# Запуск

def git(*args):
    try:
        res = subprocess.run(["git", *args], cwd=Path(__file__).parent, capture_output=True, text=True, check=True)
        return res.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def device_name(device):
    if device.type == "cuda":
        return torch.cuda.get_device_name(0)
    return f"{device.type}, {platform.machine()}"


def run_one(regime, seed, rows, data, device, split_hash):
    train_rows, val_rows = part_rows(rows, "train"), part_rows(rows, "validation")
    model = build_model(regime, seed).to(device)
    train_dl, val_dl = train_loader(train_rows, data), eval_loader(val_rows, data)
    opt = make_optimizer(model, regime)
    loss_fn = nn.CrossEntropyLoss()
    y_val = label_ids(val_rows)

    losses, f1s = [], []
    start = time.perf_counter()
    for epoch in range(1, EPOCHS + 1):
        set_train_mode(model, regime)
        total = 0.0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            total += loss.item() * len(y)
        logits = predict_logits(model, val_dl, device)
        losses.append(total / len(train_rows))
        f1s.append(macro_f1(y_val, logits))
        print(f"{regime} зерно {seed} епоха {epoch}: втрата {losses[-1]:.4f}, macro-F1 {f1s[-1]:.4f}")
    seconds = time.perf_counter() - start

    commit = git("rev-parse", "HEAD")
    dirty = None if commit is None else bool(git("status", "--porcelain", "--", "*.py", "pyproject.toml", "uv.lock"))
    config = {
        "pretrained_weights": "IMAGENET1K_V1" if REGIMES[regime]["pretrained"] else None,
        "model": "resnet18, fc = Linear(512, 6)",
        "batchnorm": {"momentum": model.bn1.momentum, "eps": model.bn1.eps},
        "optimizer": {
            "name": "AdamW",
            "groups": [{"modules": list(m), "lr": lr} for m, lr in REGIMES[regime]["groups"]],
            "defaults": opt.defaults,
        },
        "eval_modules": eval_modules(regime),
        "loss": "CrossEntropyLoss, mean",
        "train_loader": {"batch_size": BATCH, "shuffle": True, "drop_last": False, "num_workers": 0},
        "eval_loader": {"batch_size": 64, "shuffle": False, "num_workers": 0},
        "epochs": EPOCHS,
        "steps_per_epoch": len(train_dl),
        "dtype": "float32",
        "weight_decay": WEIGHT_DECAY,
        "img_size": IMG_SIZE,
        "transforms": {
            "train": [str(t) for t in TRAIN_TF.transforms],
            "eval": [str(t) for t in EVAL_TF.transforms],
        },
    }
    log = {
        "regime": regime,
        "seed": seed,
        "config": config,
        "commit": commit,
        "uncommitted_changes": dirty,
        "split_sha256": split_hash,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "device": device_name(device),
        "train_loss": losses,
        "val_f1": f1s,
        "train_seconds": seconds,
    }
    out = RUNS / f"{regime}_seed{seed}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "run.json").write_text(json.dumps(log, indent=2, default=str), encoding="utf-8")
    np.save(out / "val_logits.npy", logits.astype(np.float32))
    torch.save({k: v.cpu() for k, v in model.state_dict().items()}, out / "weights.pt")
    return f1s[-1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="папка з seg_train і seg_test")
    data = Path(parser.parse_args().data)
    device = pick_device()
    RESULTS.mkdir(exist_ok=True)

    make_split(data)
    rows = read_split()  # далі дані беруться лише з split.csv
    split_hash = file_sha256(SPLIT)
    print(f"пристрій {device_name(device)}, хеш split.csv {split_hash[:12]}")

    ok = check_split(rows)
    save_batch_figure(rows, data, RESULTS / "batch.png")
    say("\n## Початкова втрата\n")
    say(md_table(["Режим", "L0", "ln 6"], [
        [r, f"{initial_loss(r, rows, data, device):.4f}", f"{np.log(6):.4f}"] for r in ("scratch", "frozen")
    ]))
    ok &= check_regimes(device)
    ok &= check_changes(rows, data, device)
    (RESULTS / "checks.md").write_text("\n".join(REPORT) + "\n", encoding="utf-8")
    if not ok:
        sys.exit("перевірки до навчання не збіглися з очікуваним, запуски не починаю")
    print("перевірки пройдено")

    f1 = {key: run_one(*key, rows, data, device, split_hash) for key in RUN_LIST}
    regime, seed, _ = select(f1)
    state = torch.load(RUNS / f"{regime}_seed{seed}" / "weights.pt", map_location="cpu")
    torch.save(
        {"state_dict": state, "regime": regime, "seed": seed, "classes": CLASSES, "img_size": IMG_SIZE},
        MODEL,
    )
    print(f"вибрано {regime}, зерно {seed}, збережено {MODEL.name}")


if __name__ == "__main__":
    main()
