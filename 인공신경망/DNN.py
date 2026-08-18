import torch
import torch.nn as nn
import torch.optim as optim


class DNN(nn.Module):

    def __init__(
        self,
        input_size=10,
        num_classes=3
    ):
        super().__init__()

        # 입력 특징 → 64개 은닉 노드
        self.fc1 = nn.Linear(
            input_size,
            64
        )

        # 64개 Feature에 Batch Normalization
        self.bn1 = nn.BatchNorm1d(64)

        # 64 → 32
        self.fc2 = nn.Linear(
            64,
            32
        )

        # 32개 Feature에 Batch Normalization
        self.bn2 = nn.BatchNorm1d(32)

        # 최종 출력
        self.fc3 = nn.Linear(
            32,
            num_classes
        )

        # 활성화 함수
        self.relu = nn.ReLU()

        # 학습 중 30% Dropout
        self.dropout = nn.Dropout(0.3)


    def forward(self, x):

        # 1번째 은닉층
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)

        # 2번째 은닉층
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)

        # 출력층
        x = self.fc3(x)

        return x


# 모델 생성
model = DNN(
    input_size=10,
    num_classes=3
)


# 다중분류
criterion = nn.CrossEntropyLoss()


# AdamW
optimizer = optim.AdamW(
    model.parameters(),
    lr=0.001,
    weight_decay=0.01
)


# Learning Rate 감소
scheduler = optim.lr_scheduler.StepLR(
    optimizer,
    step_size=100,
    gamma=0.5
)


# ============================================================
# Training
# ============================================================

model.train()

for epoch in range(1000):

    output = model(X)

    loss = criterion(
        output,
        y
    )

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    scheduler.step()


# ============================================================
# Test
# ============================================================

model.eval()

with torch.no_grad():

    output = model(test_X)

    # 가장 큰 출력값의 클래스 선택
    prediction = torch.argmax(
        output,
        dim=1
    )

    print(prediction)