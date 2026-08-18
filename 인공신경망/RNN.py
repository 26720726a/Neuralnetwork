''' 입력 데이터 
(32, 20, 5)

32 = Batch
20 = 시간 길이
5  = 각 시간의 Feature
'''
import torch
import torch.nn as nn
import torch.optim as optim


class RNNModel(nn.Module):

    def __init__(
        self,
        input_size=5,
        hidden_size=64,
        num_classes=3
    ):
        super().__init__()

        self.rnn = nn.RNN(

            # 각 시간마다 들어오는 Feature 수
            input_size=input_size,

            # Hidden State 크기
            hidden_size=hidden_size,

            # RNN 2층
            num_layers=2,

            # 입력:
            # (batch, sequence, feature)
            batch_first=True,

            # RNN 층 사이 Dropout
            dropout=0.3
        )

        # Hidden Feature 정규화
        self.norm = nn.LayerNorm(
            hidden_size
        )

        self.dropout = nn.Dropout(0.3)

        self.fc = nn.Linear(
            hidden_size,
            num_classes
        )


    def forward(self, x):

        # output:
        # (batch, sequence, hidden_size)
        output, hidden = self.rnn(x)

        # 마지막 시간의 출력 사용
        x = output[:, -1, :]

        x = self.norm(x)

        x = self.dropout(x)

        x = self.fc(x)

        return x


model = RNNModel()


criterion = nn.CrossEntropyLoss()


optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)

loss.backward()

torch.nn.utils.clip_grad_norm_(
    model.parameters(),
    max_norm=1.0
)

optimizer.step()