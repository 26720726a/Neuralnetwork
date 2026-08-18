import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# =========================================================
# 2.1 데이터 전처리
# =========================================================
df = pd.read_csv("MSFT.csv")

# 결측치가 있는 행 제거 (dropna는 원본을 안 바꾸므로 반드시 재할당)
df = df.dropna().reset_index(drop=True)

print("===== 2.1 결측치 제거 후 데이터프레임 =====")
print(df)

# =========================================================
# 2.2 레이블링 및 슬라이싱
# =========================================================
# 학습에 쓸 6개 속성
feature_cols = ["Open", "Close", "High", "Low", "Volume", "Vwap"]

# 다음 거래일 값 = 다음 행의 값 -> shift(-1)로 한 칸 위로 당김
# (데이터가 날짜 오름차순으로 정렬돼 있다고 가정)
df["next_Close"] = df["Close"].shift(-1)
df["next_Vwap"] = df["Vwap"].shift(-1)

# 마지막 행은 "다음 날"이 없어 NaN이 되므로 제거
df = df.dropna().reset_index(drop=True)

label_cols = ["next_Close", "next_Vwap"]
data = df[feature_cols + label_cols]

print("\n===== 2.2 6개 속성 + 2개 레이블 =====")
print(data)

# numpy 배열로 변환
X = df[feature_cols].values.astype(np.float32)   # (N, 6)
Y = df[label_cols].values.astype(np.float32)     # (N, 2)

# =========================================================
# 2.5 데이터 분할 (Train:Val:Test = 6:2:2, shuffle=False)
#     -> 스케일링을 train 기준으로 해야 하므로 분할을 먼저 수행
# =========================================================
n = len(X)
n_train = int(n * 0.6)
n_val = int(n * 0.2)

X_train, Y_train = X[:n_train], Y[:n_train]
X_val,   Y_val   = X[n_train:n_train + n_val], Y[n_train:n_train + n_val]
X_test,  Y_test  = X[n_train + n_val:], Y[n_train + n_val:]

# ---- 표준화 (정규화) ----
# 주가는 몇 달러 ~ 수백 달러, 거래량은 수백만 단위로 스케일 차이가 큼.
# 스케일을 안 맞추면 학습이 거의 안 되므로 표준화 필요.
# 통계량은 반드시 train 에서만 계산 (val/test 정보 누수 방지)
x_mean, x_std = X_train.mean(axis=0), X_train.std(axis=0)
y_mean, y_std = Y_train.mean(axis=0), Y_train.std(axis=0)


def standardize(a, m, s):
    return (a - m) / s


X_train_s = standardize(X_train, x_mean, x_std)
X_val_s   = standardize(X_val,   x_mean, x_std)
X_test_s  = standardize(X_test,  x_mean, x_std)

Y_train_s = standardize(Y_train, y_mean, y_std)
Y_val_s   = standardize(Y_val,   y_mean, y_std)
# Y_test 는 나중에 원래 스케일 그대로 평가에 사용

# ---- 텐서로 변환 ----
X_train_t = torch.tensor(X_train_s, dtype=torch.float32)
Y_train_t = torch.tensor(Y_train_s, dtype=torch.float32)
X_val_t   = torch.tensor(X_val_s,   dtype=torch.float32)
Y_val_t   = torch.tensor(Y_val_s,   dtype=torch.float32)
X_test_t  = torch.tensor(X_test_s,  dtype=torch.float32)

# =========================================================
# 2.4 모델 설계 (입력 6, 출력 2, 파라미터 1000개 이하 회귀 모델)
# =========================================================
class NeuralNetwork(nn.Module):

    def __init__(self):
        super().__init__()
        # 회귀이므로 출력층에는 Sigmoid 같은 활성화를 붙이지 않음
        self.net = nn.Sequential(
            nn.Linear(6, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 2),
        )

    def forward(self, x):
        return self.net(x)


model = NeuralNetwork()

# 파라미터 수 확인 (1000개 이하인지 검증)
n_params = sum(p.numel() for p in model.parameters())
print("\n파라미터 수:", n_params)

# =========================================================
# 2.3 손실 함수 (두 레이블 MSE를 0.8 : 0.2 로 결합)
# =========================================================
def weighted_mse(pred, target):
    # pred, target : (N, 2)  -> [:,0]=next_Close, [:,1]=next_Vwap
    l1 = ((pred[:, 0] - target[:, 0]) ** 2).mean()   # 종가 손실
    l2 = ((pred[:, 1] - target[:, 1]) ** 2).mean()   # Vwap 손실
    return 0.8 * l1 + 0.2 * l2


optimizer = optim.Adam(model.parameters(), lr=0.01)

# =========================================================
# 2.5 모델 훈련 + 2.4 Early Stopping / Best 모델 저장
# =========================================================
n_epochs = 2000
patience = 20            # 검증 손실이 20 에포크 동안 개선 없으면 종료
best_val = float("inf")
wait = 0

train_losses = []
val_losses = []

for epoch in range(n_epochs):
    # ---- 훈련 ----
    model.train()
    output = model(X_train_t)
    loss = weighted_mse(output, Y_train_t)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # ---- 검증 ----
    model.eval()
    with torch.no_grad():
        val_output = model(X_val_t)
        val_loss = weighted_mse(val_output, Y_val_t)

    train_losses.append(loss.item())
    val_losses.append(val_loss.item())

    # ---- Best 모델 저장 + Early Stopping ----
    if val_loss.item() < best_val:
        best_val = val_loss.item()
        torch.save(model.state_dict(), "best_model.pth")   # 가장 좋은 모델 저장
        wait = 0
    else:
        wait += 1
        if wait >= patience:
            print(f"\nEarly stopping at epoch {epoch + 1}")
            break

    if (epoch + 1) % 50 == 0:
        print(f"epoch: {epoch + 1}  train: {loss.item():.4f}  val: {val_loss.item():.4f}")

# 가장 성능이 좋았던 모델 복원
model.load_state_dict(torch.load("best_model.pth"))

# ---- Learning Curve ----
plt.figure()
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Learning Curve")
plt.legend()
plt.show()

# =========================================================
# 2.6 모델 평가 (Test set)
# =========================================================
model.eval()
with torch.no_grad():
    pred_test_s = model(X_test_t).numpy()          # 표준화된 예측

# 예측값을 원래 스케일(달러)로 복원
pred_test = pred_test_s * y_std + y_mean
true_test = Y_test                                 # 이미 원래 스케일

# 원래 스케일 기준 MSE (해석 가능한 성능 지표)
mse_close = ((pred_test[:, 0] - true_test[:, 0]) ** 2).mean()
mse_vwap  = ((pred_test[:, 1] - true_test[:, 1]) ** 2).mean()
print(f"\n[Test] Close MSE: {mse_close:.4f}   Vwap MSE: {mse_vwap:.4f}")

# 실제값 vs 예측값 겹쳐 그리기
plt.figure()
plt.plot(true_test[:, 0], label="Actual Close")
plt.plot(pred_test[:, 0], label="Predicted Close")
plt.xlabel("Test sample")
plt.ylabel("Price")
plt.title("Next-day Close: Actual vs Predicted")
plt.legend()
plt.show()

plt.figure()
plt.plot(true_test[:, 1], label="Actual Vwap")
plt.plot(pred_test[:, 1], label="Predicted Vwap")
plt.xlabel("Test sample")
plt.ylabel("Price")
plt.title("Next-day Vwap: Actual vs Predicted")
plt.legend()
plt.show()