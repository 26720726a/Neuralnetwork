import torch
import torch.nn as nn
import torch.optim as optim

X = torch.tensor([
    [0.1, 0.2, 0.1],
    [0.2, 0.4, 0.3],
    [0.4, 0.5, 0.4],
    [0.6, 0.7, 0.8],
    [0.8, 0.9, 0.7],
    [0.9, 0.8, 0.9]
], dtype=torch.float32)

y = torch.tensor([
    [0.0],
    [0.0],
    [0.0],
    [1.0],
    [1.0],
    [1.0]
], dtype=torch.float32)

class NeuralNetwork(nn.Module):
    def __init__(self):
            super().__init__()

            self.fc1=nn.Linear(3,5)

            self.relu=nn.ReLU()

            self.fc2=nn.Linear(5,1)

            self.sigmoid=nn.Sigmoid()

    def forward(self, x):
          x=self.fc1(x)

          x=self.relu(x)

          x=self.fc2(x)

          x=self.sigmoid(x)

          return x

model = NeuralNetwork()

criterion = nn.BCELoss()

optimizer = optim.SGD(
    model.parameters(),
    lr=0.1
)

for epoch in range(2000):
    output=model(X)

    loss=criterion(output,y)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    if (epoch+1) % 200 == 0:
        print("epoch:",epoch + 1,"loss:",loss.item())

test_X = torch.tensor([
    [0.3, 0.3, 0.2],
    [0.7, 0.8, 0.9]
], dtype=torch.float32)

model.eval()

with torch.no_grad():
    output=model(test_X)

    prediction = (output >= 0.5).float()
      
    print("출력 확률:", output)
    print("예측:", prediction)