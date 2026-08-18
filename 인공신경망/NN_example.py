import torch
import torch.nn as nn
import torch.optim as optim

X = torch.tensor([
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0]
], dtype=torch.float32)

y = torch.tensor([
    [0.0],
    [1.0],
    [1.0],
    [1.0]
], dtype=torch.float32)

class NeuralNetwork(nn.Module):

    def __init__(self):
        super().__init__()

        # 입력 2개 → 은닉노드 4개
        self.fc1 = nn.Linear(2, 4)

        # 활성화 함수
        self.relu = nn.ReLU()

        # 출력층 활성화 함수 
        self.sigmoid=nn.Sigmoid()

        # 은닉노드 4개 → 출력 1개
        self.fc2 = nn.Linear(4, 1)

    def forward(self, x):

        x = self.fc1(x)

        x = self.relu(x)

        x = self.fc2(x)

        x = self.sigmoid(x)

        return x

# 3. 모델 생성
model = NeuralNetwork()

# 4. Loss 함수
criterion = nn.BCELoss()

# 5. Optimizer
optimizer = optim.SGD(
    model.parameters(),
    lr=0.1
)

# 6. 학습
for epoch in range(1000):

    output=model(X)

    loss = criterion(output, y)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    if (epoch + 1) % 100 == 0:
    
        print(
            "epoch:",
            epoch + 1,
            "loss:",
            loss.item()
        )

model.eval() # 모델을 평가로 변경 

with torch.no_grad(): # gradient 계산 안함 

    output = model(X)

    prediction = (output >= 0.5).float()

    print("예측:", prediction)
    print("정답:", y)