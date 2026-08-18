import torch
import torch.nn as nn
import torch.optim as optim


# =====================================
# 1. 데이터 준비
# =====================================

X = torch.tensor([
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0],
    [2.0, 0.0],
    [0.0, 2.0]
], dtype=torch.float32)

# 3개의 클래스: 0, 1, 2
y = torch.tensor([
    0,
    1,
    1,
    2,
    2,
    2
], dtype=torch.long)


# =====================================
# 2. 신경망 정의
# =====================================

class NeuralNetwork(nn.Module):

    def __init__(self):
        super().__init__()

        # 입력 2개 → 은닉노드 8개
        self.fc1 = nn.Linear(2, 8)

        # 활성화 함수
        self.relu = nn.ReLU()

        # 은닉노드 8개 → 출력 3개
        self.fc2 = nn.Linear(8, 3)


    def forward(self, x):

        x = self.fc1(x)

        x = self.relu(x)

        x = self.fc2(x)

        return x

# =====================================
# 3. 모델 생성
# =====================================
model = NeuralNetwork()

# =====================================
# 4. Loss 함수
# =====================================
criterion = nn.CrossEntropyLoss()

# =====================================
# 5. Optimizer
# =====================================

optimizer = optim.SGD(
    model.parameters(),
    lr=0.1
)


# =====================================
# 6. 학습
# =====================================

for epoch in range(100):

    # ---------------------------
    # Forward
    # ---------------------------

    output = model(X)

    loss = criterion(output, y)

    # ---------------------------
    # 이전 gradient 제거, 초기화
    # ---------------------------
    optimizer.zero_grad()

    # ---------------------------
    # Backpropagation, gradient 계산
    # ---------------------------
    loss.backward()

    # ---------------------------
    # Gradient Descent
    # ---------------------------
    optimizer.step()

    # ---------------------------
    # 중간 결과 확인
    # ---------------------------

    if (epoch + 1) % 100 == 0:

        print(
            "epoch:",
            epoch + 1,
            "loss:",
            loss.item()
        )

# =====================================
# 7. 학습 완료 후 예측
# =====================================

model.eval()

with torch.no_grad():

    output = model(X)

    prediction = torch.argmax(
        output,
        dim=1
    )

    print("예측:", prediction)
    print("정답:", y)