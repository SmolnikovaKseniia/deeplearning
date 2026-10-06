import sys
import numpy as np
import torch
from sklearn.datasets import load_iris


def load():
    X, y = load_iris(return_X_y=True)
    rng = np.random.default_rng(0)
    train, test = [], []
    for c in range(3):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        train += list(idx[:35])
        test += list(idx[35:])
    mean = X[train].mean(axis=0)
    std = X[train].std(axis=0, ddof=0)
    return (X[train] - mean) / std, y[train], (X[test] - mean) / std, y[test]


def init():
    rng = np.random.default_rng(0)
    W1 = rng.normal(0, np.sqrt(2 / 4), (4, 8))
    W2 = rng.normal(0, np.sqrt(2 / (8 + 3)), (8, 3))
    return {"W1": W1, "b1": np.zeros(8), "W2": W2, "b2": np.zeros(3)}


def forward(P, X, y):
    assert y.shape == (X.shape[0],)
    z1 = X @ P["W1"] + P["b1"]
    a1 = np.maximum(z1, 0)
    z2 = a1 @ P["W2"] + P["b2"]
    assert z1.shape == (X.shape[0], 8) and z2.shape == (X.shape[0], 3)
    s = z2 - z2.max(axis=1, keepdims=True)
    log_q = s - np.log(np.exp(s).sum(axis=1, keepdims=True))
    loss = -log_q[np.arange(len(y)), y].mean()
    return loss, (X, z1, a1, np.exp(log_q), y)


def backward(P, cache, bug=False):
    X, z1, a1, q, y = cache
    N = len(y)
    dz2 = q.copy()
    dz2[np.arange(N), y] -= 1
    if not bug:
        dz2 = dz2 / N
    dW2 = a1.T @ dz2
    db2 = dz2.sum(axis=0)
    dz1 = (dz2 @ P["W2"].T) * (z1 > 0)
    dW1 = X.T @ dz1
    db1 = dz1.sum(axis=0)
    grads = {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}
    for k in grads:
        assert grads[k].shape == P[k].shape, k
    return grads


def torch_ref(P, X, y):
    net = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.ReLU(), torch.nn.Linear(8, 3)).double()
    with torch.no_grad():
        net[0].weight.copy_(torch.from_numpy(P["W1"].T))
        net[0].bias.copy_(torch.from_numpy(P["b1"]))
        net[2].weight.copy_(torch.from_numpy(P["W2"].T))
        net[2].bias.copy_(torch.from_numpy(P["b2"]))
    loss = torch.nn.functional.cross_entropy(net(torch.from_numpy(X)), torch.from_numpy(y))
    loss.backward()
    g = {"W1": net[0].weight.grad.numpy().T, "b1": net[0].bias.grad.numpy(),
         "W2": net[2].weight.grad.numpy().T, "b2": net[2].bias.grad.numpy()}
    return loss.item(), g


def numeric_grad(P, X, y, name, i, eps=1e-6):
    old = P[name][i]
    P[name][i] = old + eps
    loss_plus = forward(P, X, y)[0]
    P[name][i] = old - eps
    loss_minus = forward(P, X, y)[0]
    P[name][i] = old
    return (loss_plus - loss_minus) / (2 * eps)


def main(bug=False):
    X, y, _, _ = load()
    P = init()
    loss, cache = forward(P, X, y)
    g = backward(P, cache, bug)
    torch_loss, torch_g = torch_ref(P, X, y)

    print("bug =", bug)
    print("loss numpy =", loss, "| torch =", torch_loss)
    print("\n| Величина | max abs diff | OK |\n")
    d = abs(loss - torch_loss)
    print(f"| Втрата | {d:.3e} | {d <= 1e-12} |")
    for name in g:
        d = np.abs(g[name] - torch_g[name]).max()
        print(f"| grad {name} | {d:.3e} | {d <= 1e-12} |")

    print("\n| Параметр | backward() | чисельна | abs diff | OK |\n")
    nums = []
    for name, i in [("W1", (0, 0)), ("b1", 0), ("W2", (0, 0)), ("b2", 0)]:
        num = numeric_grad(P, X, y, name, i)
        man = g[name][i]
        nums.append(num)
        d = abs(num - man)
        print(f"| {name}[{i}] | {man:.10e} | {num:.10e} | {d:.3e} | {d <= 1e-7} |")

    # за умовою всі порівнювані значення мають бути скінченними
    finite = (np.isfinite([loss, torch_loss]).all()
              and all(np.isfinite(g[k]).all() and np.isfinite(torch_g[k]).all() for k in g)
              and np.isfinite(nums).all())
    print("\nусі значення скінченні:", bool(finite))


if __name__ == "__main__":
    main(bug="--bug" in sys.argv)
