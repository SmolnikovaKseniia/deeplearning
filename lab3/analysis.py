"""Аналіз: таблиця запусків, криві, розклад ефектів, вибір, одноразова оцінка на test.

Запуск: uv run python analysis.py --data path/to/intel
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # без вікон, працює на будь-якій ОС
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from PIL import Image
from sklearn.metrics import confusion_matrix
from torch import nn

from common import (
    CLASSES, MODEL, RESULTS, RUN_LIST, RUNS, SPLIT, eval_loader, file_sha256, image_path,
    label_ids, macro_f1, md_table, part_rows, predict_logits, read_split, select,
)

COLORS = {"scratch": "#4C78A8", "frozen": "#F58518", "unfreeze": "#54A24B"}
STYLES = {1: "-", 2: "--", 3: ":"}


# Журнали

def load_runs():
    runs = {}
    for regime, seed in RUN_LIST:
        path = RUNS / f"{regime}_seed{seed}" / "run.json"
        if not path.exists():
            sys.exit(f"немає журналу {path}")
        runs[(regime, seed)] = json.loads(path.read_text(encoding="utf-8"))
    return runs


def check_runs(runs, rows):
    """Хеш split.csv і збіг F1 з логітів з журналом."""
    digest = file_sha256(SPLIT)
    bad = [f"{r}_{s}" for (r, s), log in runs.items() if log["split_sha256"] != digest]
    if bad:
        sys.exit(f"хеш split.csv не збігається з журналами ({', '.join(bad)}): перевірте поділ і повторіть навчання всіх семи запусків")
    y_val = label_ids(part_rows(rows, "validation"))
    for (regime, seed), log in runs.items():
        logits = np.load(RUNS / f"{regime}_seed{seed}" / "val_logits.npy")
        if abs(macro_f1(y_val, logits) - log["val_f1"][-1]) > 1e-9:
            sys.exit(f"val_logits.npy не збігається з журналом: {regime}, зерно {seed}")


# Таблиці й рисунки

def run_table(runs):
    return [
        [i + 1, regime, seed, f"{log['train_loss'][-1]:.4f}", f"{log['val_f1'][-1]:.4f}", f"{log['train_seconds']:.0f}"]
        for i, ((regime, seed), log) in enumerate(runs.items())
    ]


def plot_curves(runs, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True)
    for (regime, seed), log in runs.items():
        epochs = range(1, len(log["val_f1"]) + 1)
        for ax, key in zip(axes, ("train_loss", "val_f1")):
            ax.plot(epochs, log[key], color=COLORS[regime], linestyle=STYLES[seed], label=f"{regime} {seed}")
    axes[0].set(title="Втрата на train", xlabel="епоха", ylabel="втрата")
    axes[1].set(title="macro-F1 на validation", xlabel="епоха", ylabel="macro-F1")
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def effects(f1):
    """Середнє й розмах за зернами, поріг шуму, ефекти Δpre і Δunf."""
    seeds = (1, 2, 3)
    mean = {r: float(np.mean([f1[(r, s)] for s in seeds])) for r in ("frozen", "unfreeze")}
    spread = {r: max(f1[(r, s)] for s in seeds) - min(f1[(r, s)] for s in seeds) for r in mean}
    noise = max(spread.values())
    return {
        "mean": mean,
        "spread": spread,
        "R": noise,
        "d_pre": mean["frozen"] - f1[("scratch", 1)],
        "d_unf": mean["unfreeze"] - mean["frozen"],
    }


def verdict(delta, noise):
    return "не відрізняється від розкиду" if abs(delta) <= noise else "вартий перевірки, але не доведений"


# Test

def load_selected(f1):
    """Перевіряє, що lab3_model.pt відповідає правилу вибору за журналами."""
    regime, seed, mean = select(f1)
    ckpt = torch.load(MODEL, map_location="cpu")
    if (ckpt["regime"], ckpt["seed"]) != (regime, seed):
        sys.exit(f"lab3_model.pt: {ckpt['regime']}, зерно {ckpt['seed']}; за журналами має бути {regime}, зерно {seed}")
    return regime, seed, mean, ckpt


def evaluate_test(ckpt, test_rows, data):
    """Єдина оцінка на test, на CPU, у порядку split.csv."""
    model = torchvision.models.resnet18(weights=None)
    model.fc = nn.Linear(512, len(CLASSES))
    model.load_state_dict(ckpt["state_dict"])
    return predict_logits(model, eval_loader(test_rows, data), torch.device("cpu"))


def typical_errors(cm, y, pred, rows, logits, n=5):
    """n найчастіших пар (справжній, передбачений); файл першим за іменем як рядок."""
    pairs = sorted(
        [(t, p) for t in range(len(CLASSES)) for p in range(len(CLASSES)) if t != p and cm[t, p] > 0],
        key=lambda tp: (-cm[tp], tp[0], tp[1]),
    )
    if not pairs:
        return []
    members = {
        (t, p): sorted(rows[i]["file"] for i in range(len(rows)) if y[i] == t and pred[i] == p)
        for t, p in pairs
    }
    index = {r["file"]: i for i, r in enumerate(rows)}
    probs = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
    out = []
    for rank in range(max(len(m) for m in members.values())):  # другий файл пари, якщо пар менше n
        for t, p in pairs:
            if len(out) < n and rank < len(members[(t, p)]):
                name = members[(t, p)][rank]
                out.append({"file": name, "true": t, "pred": p, "prob": probs[index[name], p], "count": cm[t, p]})
    return out


def plot_confusion(cm, path):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(cm, cmap="Blues")
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set(xticks=range(len(CLASSES)), yticks=range(len(CLASSES)), xlabel="передбачений клас", ylabel="справжній клас", title="Матриця плутанини, test")
    ax.set_xticklabels(CLASSES, rotation=45, ha="right")
    ax.set_yticklabels(CLASSES)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_errors(errors, data, path):
    fig, axes = plt.subplots(1, len(errors), figsize=(3.2 * len(errors), 3.6))
    for ax, e in zip(np.atleast_1d(axes), errors):
        row = {"part": "test", "label": CLASSES[e["true"]], "file": e["file"]}
        ax.imshow(Image.open(image_path(data, row)).convert("RGB"))
        ax.set_title(f"{e['file']}\n{CLASSES[e['true']]} → {CLASSES[e['pred']]}\np = {e['prob']:.2f}", fontsize=9)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="папка з seg_train і seg_test")
    data = Path(parser.parse_args().data)
    RESULTS.mkdir(exist_ok=True)

    rows = read_split()
    runs = load_runs()
    check_runs(runs, rows)
    print(f"журнали: {len(runs)}, хеш split.csv збігається")

    f1 = {key: log["val_f1"][-1] for key, log in runs.items()}
    report = ["# Звіт\n", "## Сім запусків\n"]
    report.append(md_table(["#", "Режим", "Зерно", "Втрата train, остання епоха", "macro-F1 validation, остання епоха", "Час навчання, с"], run_table(runs)))
    plot_curves(runs, RESULTS / "curves.png")

    e = effects(f1)
    report += ["\n## Розклад ефектів\n", md_table(["Величина", "Значення"], [
        ["F̄_frozen, R_frozen", f"{e['mean']['frozen']:.4f}, {e['spread']['frozen']:.4f}"],
        ["F̄_unfreeze, R_unfreeze", f"{e['mean']['unfreeze']:.4f}, {e['spread']['unfreeze']:.4f}"],
        ["R", f"{e['R']:.4f}"],
        ["Δ_pre", f"{e['d_pre']:+.4f}: {verdict(e['d_pre'], e['R'])}"],
        ["Δ_unf", f"{e['d_unf']:+.4f}: {verdict(e['d_unf'], e['R'])}"],
    ])]
    report.append(f"\nΔ_pre + Δ_unf = {e['d_pre'] + e['d_unf']:+.4f} (повний виграш трансферу відносно scratch, зерно 1).")
    times = {r: np.mean([log["train_seconds"] for (rr, _), log in runs.items() if rr == r]) for r in COLORS}
    report.append("\nСередній час навчання, с: " + ", ".join(f"{r} {t:.0f}" for r, t in times.items()) + ".")
    print(f"R {e['R']:.4f}, Δ_pre {e['d_pre']:+.4f} ({verdict(e['d_pre'], e['R'])}), Δ_unf {e['d_unf']:+.4f} ({verdict(e['d_unf'], e['R'])})")

    regime, seed, mean, ckpt = load_selected(f1)
    note = "Вибір між frozen і unfreeze визначив шум (|Δ_unf| ≤ R)." if abs(e["d_unf"]) <= e["R"] and regime != "scratch" else ""
    report += [
        "\n## Вибір за validation\n",
        "Середнє F за режимом: " + ", ".join(f"{r} {m:.4f}" for r, m in mean.items()) + ".",
        f"\nВибрано: {regime}, зерно {seed}, F = {f1[(regime, seed)]:.4f}. lab3_model.pt відповідає правилу вибору. {note}",
    ]
    print(f"вибрано {regime}, зерно {seed}, validation F {f1[(regime, seed)]:.4f}")

    # Test обчислюється один раз, лише після всіх перевірок вище
    test_rows = part_rows(rows, "test")
    y = label_ids(test_rows)
    logits = evaluate_test(ckpt, test_rows, data)
    pred = logits.argmax(1)
    f_test = macro_f1(y, logits)
    p = float((pred == y).mean())
    se = float(np.sqrt(p * (1 - p) / len(y)))
    cm = confusion_matrix(y, pred, labels=list(range(len(CLASSES))))
    report += [
        "\n## Test\n",
        f"macro-F1 на test: {f_test:.4f}; на validation: {f1[(regime, seed)]:.4f}; різниця test − validation: {f_test - f1[(regime, seed)]:+.4f}.",
        f"\nAccuracy {p:.4f}, n = {len(y)}, стандартна похибка √(p(1−p)/n) = {se:.4f}.",
        "\n### Матриця плутанини (рядки — справжній клас, стовпці — передбачений)\n",
        md_table(["", *CLASSES], [[c, *row] for c, row in zip(CLASSES, cm.tolist())]),
    ]
    plot_confusion(cm, RESULTS / "confusion.png")
    print(f"test: macro-F1 {f_test:.4f}, accuracy {p:.4f}, стандартна похибка {se:.4f}")

    errors = typical_errors(cm, y, pred, test_rows, logits)
    if errors:
        plot_errors(errors, data, RESULTS / "errors.png")
    report += ["\n### П'ять характерних помилок\n", md_table(
        ["Файл", "Справжній клас", "Передбачений клас", "Імовірність передбаченого класу", "Кількість таких помилок у test"],
        [[x["file"], CLASSES[x["true"]], CLASSES[x["pred"]], f"{x['prob']:.4f}", x["count"]] for x in errors],
    )]
    (RESULTS / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"збережено в {RESULTS.name}/")


if __name__ == "__main__":
    main()
