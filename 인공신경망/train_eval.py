"""
train_eval.py  --  PyTorch 학습 파이프라인 모듈

한 번 호출하면 아래를 모두 처리:
  1) train / validation 분할
  2) Early Stopping (검증 손실 기준)
  3) 가장 좋은 모델 저장 + 복원
  4) Learning Curve 시각화
  5) (분류) 정확도 평가

사용법:
    from train_eval import train_model, evaluate

    history, model = train_model(
        model, criterion, optimizer,
        X, y,                      # 전체 데이터 -> 내부에서 train/val 분할
        task="classification",     # 또는 "regression"
        epochs=1000, patience=20,
    )
    acc = evaluate(model, test_X, test_y, task="classification")
"""

import copy
import torch
import matplotlib.pyplot as plt


# =========================================================
# train / validation 분할
# =========================================================
def train_val_split(X, y, val_ratio=0.2, shuffle=True, seed=42):
    """전체 데이터를 train/val 로 나눔.
    시계열(순서 유지 필요)이면 shuffle=False 로 호출."""
    n = X.shape[0]
    n_val = int(n * val_ratio)

    if shuffle:
        g = torch.Generator().manual_seed(seed)
        idx = torch.randperm(n, generator=g)
    else:
        idx = torch.arange(n)          # 순서 유지 (시계열)

    # 앞쪽 = train, 뒤쪽 = val
    # (시계열이면 과거로 학습하고 최신 구간을 검증에 사용)
    train_idx = idx[:n - n_val]
    val_idx = idx[n - n_val:]
    return X[train_idx], y[train_idx], X[val_idx], y[val_idx]


# 내부용: 미니배치 순회 (batch_size=None 이면 전체 배치)
def _iterate_batches(X, y, batch_size, shuffle, device):
    n = X.shape[0]
    if batch_size is None or batch_size >= n:
        yield X.to(device), y.to(device)
        return
    order = torch.randperm(n) if shuffle else torch.arange(n)
    for i in range(0, n, batch_size):
        b = order[i:i + batch_size]
        if b.numel() == 1:      # 크기 1 배치는 BatchNorm 에러 방지로 건너뜀
            continue
        yield X[b].to(device), y[b].to(device)


# =========================================================
# 학습 (분할 + Early Stopping + best 저장 + Learning Curve)
# =========================================================
def train_model(model, criterion, optimizer,
                X, y,
                X_val=None, y_val=None,       # 이미 분할했으면 여기로 전달
                val_ratio=0.2, split_shuffle=True, seed=42,
                epochs=1000, patience=20, batch_size=None,
                task="classification",        # "classification" | "regression"
                save_path="best_model.pth",
                plot=True, verbose=True, device=None):

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # 1) train/val 분할 (val 데이터가 안 주어졌을 때만 내부 분할)
    if X_val is None:
        X_train, y_train, X_val, y_val = train_val_split(
            X, y, val_ratio, split_shuffle, seed)
    else:
        X_train, y_train = X, y

    X_val_d, y_val_d = X_val.to(device), y_val.to(device)

    best_val = float("inf")
    best_state = None
    wait = 0
    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    for epoch in range(epochs):
        # ---- 훈련 ----
        model.train()
        running, n_seen = 0.0, 0
        for xb, yb in _iterate_batches(X_train, y_train, batch_size, True, device):
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            running += loss.item() * xb.shape[0]
            n_seen += xb.shape[0]
        train_loss = running / n_seen

        # ---- 검증 ----
        model.eval()
        with torch.no_grad():
            val_out = model(X_val_d)
            val_loss = criterion(val_out, y_val_d).item()
            val_acc = None
            if task == "classification":
                pred = val_out.argmax(dim=1)
                val_acc = (pred == y_val_d).float().mean().item()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # ---- best 저장 + Early Stopping ----
        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            torch.save(best_state, save_path)      # 파일로도 저장
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch + 1}")
                break

        if verbose and (epoch + 1) % 50 == 0:
            msg = f"epoch {epoch+1}  train {train_loss:.4f}  val {val_loss:.4f}"
            if val_acc is not None:
                msg += f"  val_acc {val_acc:.4f}"
            print(msg)

    # 가장 좋았던 가중치 복원
    if best_state is not None:
        model.load_state_dict(best_state)

    # ---- Learning Curve ----
    if plot:
        plt.figure()
        plt.plot(history["train_loss"], label="Train Loss")
        plt.plot(history["val_loss"], label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Learning Curve")
        plt.legend()
        plt.show()

    return history, model


# =========================================================
# 평가 (test set)
# =========================================================
@torch.no_grad()
def evaluate(model, X, y, task="classification", criterion=None, device=None):
    """분류면 정확도(accuracy), 회귀면 손실(loss) 반환."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    out = model(X.to(device))
    y = y.to(device)

    if task == "classification":
        pred = out.argmax(dim=1)
        return (pred == y).float().mean().item()      # accuracy
    else:
        if criterion is None:
            criterion = torch.nn.MSELoss()
        return criterion(out, y).item()               # loss
