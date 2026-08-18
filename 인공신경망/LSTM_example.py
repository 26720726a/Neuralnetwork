import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input

# =========================================================
# 2.1 데이터 준비
# =========================================================
X = np.array([
    [0, 1, 2],
    [1, 2, 3],
    [2, 3, 4],
    [3, 4, 5],
    [4, 5, 6],
    [5, 6, 7],
    [6, 7, 8],
    [7, 8, 9],
    [8, 9, 10],
    [9, 10, 11],
], dtype=np.float32)

y = np.array([3, 4, 5, 6, 7, 8, 9, 10, 11, 12], dtype=np.float32)

# --- 정규화 ---
# 예측해야 할 13은 훈련 타깃(최대 12) 범위 밖 값이라, 값을 그대로 쓰면
# 신경망이 외삽에 약해 12 근처에서 멈춤. 작은 값 범위로 정규화하면
# LSTM이 "다음 = 직전 + 1" 관계를 훨씬 안정적으로 학습함.
SCALE = 15.0
Xs = X / SCALE
ys = y / SCALE

# LSTM 입력 형식: (배치 크기, 시퀀스 길이, feature 수) = (10, 3, 1)
Xs = Xs.reshape(Xs.shape[0], Xs.shape[1], 1)
print("X shape:", Xs.shape)   # (10, 3, 1)
print("y shape:", ys.shape)   # (10,)

# =========================================================
# 2.2 모델 설계
# =========================================================
model = Sequential([
    Input(shape=(3, 1)),      # 시퀀스 길이 3, feature 1
    LSTM(64),                 # hidden state 차원 64
    Dense(1),                 # 최종 예측값(실수 1개)
])

model.summary()

# =========================================================
# 2.3 모델 훈련
# =========================================================
# 다음 숫자(실수)를 맞히는 회귀 -> 손실함수 MSE, optimizer adam
model.compile(optimizer="adam", loss="mse")
model.fit(Xs, ys, epochs=1000, verbose=0)

# =========================================================
# 2.4 모델 평가
# =========================================================
test_seq = np.array([[10, 11, 12]], dtype=np.float32)
test_seq_s = (test_seq / SCALE).reshape(1, 3, 1)

pred_s = model.predict(test_seq_s, verbose=0)
pred = pred_s[0, 0] * SCALE          # 정규화 되돌리기

print("입력 [10, 11, 12] -> 예측값:", pred)