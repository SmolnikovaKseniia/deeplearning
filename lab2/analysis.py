"""ЛР2, пункти 2–4: вибір порога на validation, перевірка пошуку, оцінка на test.

    uv run python analysis.py --saved results

Працює лише зі збереженими прогнозами (results/predictions.npz), без навчання і CSV.
Виводить і зберігає в results/: candidates.csv, curves.png, threshold.json,
errors.csv, report.md (усі таблиці для README).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_auc_score, roc_curve

from deeplearning.lab2.lab2_starter import load_predictions, split_summary

COLS = ["threshold", "TP", "FP", "FN", "TN", "C"]

# Набори A/B/C з умови: (оцінки, мітки).
CHECKS = {
    "A": ([0.9, 0.7, 0.7, 0.4, 0.2, 0.1], [0, 1, 0, 1, 0, 0]),
    "B": ([0.9, 0.6, 0.3], [0, 0, 0]),
    "C": ([0.8] + [0.5] * 11 + [0.2], [1, 1] + [0] * 11),
}


def confusion(y: np.ndarray, s: np.ndarray, t: float) -> tuple[int, int, int, int]:
    """Пряме застосування правила: позитивний прогноз, якщо s >= t."""
    pred = s >= t
    pos = y == 1
    tp = int(np.sum(pred & pos))
    fp = int(np.sum(pred & ~pos))
    fn = int(np.sum(~pred & pos))
    tn = int(np.sum(~pred & ~pos))
    return tp, fp, fn, tn


def cost(fn: int, fp: int) -> int:
    return 10 * fn + fp


def candidate_table(y: np.ndarray, s: np.ndarray) -> pd.DataFrame:
    """Усі кандидати (унікальні оцінки та +inf) за O(n log n): сортування + cumsum.

    Рядки йдуть за спаданням порога: +inf, потім унікальні оцінки від найбільшої.
    """
    order = np.argsort(-s, kind="stable")
    s_sorted, y_sorted = s[order], y[order]
    # Останній індекс кожної групи рівних оцінок: рівні оцінки отримують однаковий прогноз.
    last = np.r_[np.flatnonzero(s_sorted[1:] != s_sorted[:-1]), len(s) - 1]
    tp = np.cumsum(y_sorted)[last]
    fp = (last + 1) - tp
    p, n = int(y.sum()), int(len(y) - y.sum())
    tp = np.r_[0, tp]
    fp = np.r_[0, fp]
    fn, tn = p - tp, n - fp
    return pd.DataFrame(
        {"threshold": np.r_[np.inf, s_sorted[last]], "TP": tp, "FP": fp, "FN": fn, "TN": tn,
         "C": 10 * fn + fp}
    )


def best_row(table: pd.DataFrame) -> pd.Series:
    """Мінімум C; за рівної вартості — найбільший поріг (перший у таблиці за спаданням)."""
    return table.iloc[int(table["C"].to_numpy().argmin())]


def brute_force(y: np.ndarray, s: np.ndarray) -> tuple[pd.DataFrame, float]:
    """Незалежна перевірка: окремий цикл по кандидатах, без cumsum і сортування міток."""
    rows, best_t, best_c = [], None, None
    for t in [np.inf, *sorted(set(s.tolist()), reverse=True)]:
        tp, fp, fn, tn = confusion(y, s, t)
        c = cost(fn, fp)
        rows.append((t, tp, fp, fn, tn, c))
        if best_c is None or c < best_c or (c == best_c and t > best_t):
            best_t, best_c = t, c
    return pd.DataFrame(rows, columns=COLS), best_t


def fmt(v: object) -> str:
    return f"{v:.6g}" if isinstance(v, (float, np.floating)) else str(v)


def md(df: pd.DataFrame) -> str:
    head = "| " + " | ".join(map(str, df.columns)) + " |\n|" + "---|" * len(df.columns)
    body = ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([head, *body])


def rule_row(name: str, y: np.ndarray, s: np.ndarray, t: float) -> dict[str, object]:
    tp, fp, fn, tn = confusion(y, s, t)
    return {"rule": name, "threshold": t, "TP": tp, "FP": fp, "FN": fn, "TN": tn, "C": cost(fn, fp)}


def plot_curves(y: np.ndarray, s: np.ndarray, t_star: float, path: Path) -> float:
    auc = roc_auc_score(y, s)
    prec, rec, _ = precision_recall_curve(y, s)
    fpr, tpr, _ = roc_curve(y, s)
    fig, (ax_pr, ax_roc) = plt.subplots(1, 2, figsize=(10, 4))
    ax_pr.plot(rec, prec)
    ax_pr.set(xlabel="Recall", ylabel="Precision", title="PR (validation)")
    ax_roc.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
    ax_roc.plot([0, 1], [0, 1], "--", color="gray")
    ax_roc.set(xlabel="FPR", ylabel="TPR", title="ROC (validation)")
    for name, t in (("0.5", 0.5), ("t*", t_star)):
        tp, fp, fn, tn = confusion(y, s, t)
        if tp + fp:
            ax_pr.plot(tp / (tp + fn), tp / (tp + fp), "o", label=f"t = {name}")
        ax_roc.plot(fp / (fp + tn), tp / (tp + fn), "o", label=f"t = {name}")
    ax_pr.legend()
    ax_roc.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return auc


def run_checks() -> tuple[bool, list[str]]:
    ok_all, lines = True, []
    for name, (s_list, y_list) in CHECKS.items():
        s, y = np.array(s_list, dtype=np.float64), np.array(y_list, dtype=np.int64)
        fast = candidate_table(y, s)
        slow, t_slow = brute_force(y, s)
        same = np.array_equal(fast.to_numpy(dtype=np.float64), slow.to_numpy(dtype=np.float64))
        t_fast = float(best_row(fast)["threshold"])
        ok = same and t_fast == t_slow
        ok_all &= ok
        lines += [f"### Набір {name}", md(fast),
                  f"\nТаблиці збігаються: {same}; t* (cumsum) = {fmt(t_fast)}, "
                  f"t* (перебір) = {fmt(t_slow)}; C = {int(best_row(fast)['C'])}\n"]
    return ok_all, lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--saved", type=Path, default=Path("results"), help="Тека з прогнозами")
    out = parser.parse_args().saved
    data = load_predictions(out)
    report: list[str] = ["## Поділ даних", md(split_summary(data))]

    # --- Пункт 2: validation ---
    yv = data["validation_y"].astype(np.int64)
    sv = data["validation_scores"].astype(np.float64)
    table = candidate_table(yv, sv)
    table.to_csv(out / "candidates.csv", index=False)
    t_star = float(best_row(table)["threshold"])
    auc = plot_curves(yv, sv, t_star, out / "curves.png")
    rules = pd.DataFrame(
        [rule_row("Стандартний поріг", yv, sv, 0.5),
         rule_row("Мінімум вартості", yv, sv, t_star),
         rule_row("Усі негативні", yv, sv, np.inf)]
    )
    report += [f"## Validation\n\nROC-AUC = {auc:.6f}; кандидатів: {len(table)}\n", md(rules)]

    # перевірка пошуку, до звернення до test
    ok, lines = run_checks()
    report += ["## Перевірка пошуку порога", *lines]
    if not ok:
        print("\n".join(report))
        raise SystemExit("Перевірка пошуку не пройдена: test не оцінюється.")

    # test із зафіксованим порогом
    (out / "threshold.json").write_text(json.dumps({"t_star": repr(t_star)}) + "\n")
    yt = data["test_y"].astype(np.int64)
    st = data["test_scores"].astype(np.float64)
    tp, fp, fn, tn = confusion(yt, st, t_star)
    precision = tp / (tp + fp) if tp + fp else float("nan")  # не визначено: прогнозів «1» немає
    recall = tp / (tp + fn) if tp + fn else float("nan")  # не визначено: у test немає шахрайств
    test = pd.DataFrame([{"threshold": t_star, "TP": tp, "FP": fp, "FN": fn, "TN": tn,
                          "C": cost(fn, fp), "precision": precision, "recall": recall}])
    report += ["## Test", md(test)]

    ids, amount = data["test_row_ids"], data["test_amount"]
    pred = st >= t_star
    errors = []
    for kind, mask in (("FP", pred & (yt == 0)), ("FN", ~pred & (yt == 1))):
        for i in np.flatnonzero(mask)[np.argsort(ids[mask])][:5]:
            errors.append({"csv_row": int(ids[i]), "type": kind, "label": int(yt[i]),
                           "score": float(st[i]), "amount": float(amount[i]),
                           "score_minus_t": float(st[i] - t_star)})
    errors = pd.DataFrame(errors)
    errors.to_csv(out / "errors.csv", index=False)
    report += ["## Помилки (перші 5 FP і FN за номером рядка CSV)", md(errors)]

    text = "\n\n".join(report) + "\n"
    (out / "report.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
