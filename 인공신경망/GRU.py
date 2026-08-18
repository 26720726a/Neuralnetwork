import torch
import torch.nn as nn
import torch.optim as optim


class GRUModel(nn.Module):

    def __init__(
        self,
        input_size=5,
        hidden_size=64,
        num_classes=3
    ):
        super().__init__()

        self.gru = nn.GRU(

            input_size=input_size,

            hidden_size=hidden_size,

            num_layers=2,

            batch_first=True,

            dropout=0.3
        )

        self.norm = nn.LayerNorm(
            hidden_size
        )

        self.dropout = nn.Dropout(0.3)

        self.fc = nn.Linear(
            hidden_size,
            num_classes
        )


    def forward(self, x):

        # GRU는 Cell State가 없음
        output, hidden = self.gru(x)

        # 마지막 timestep
        x = output[:, -1, :]

        # 정규화
        x = self.norm(x)

        # Dropout
        x = self.dropout(x)

        # 출력
        x = self.fc(x)

        return x


model = GRUModel()


criterion = nn.CrossEntropyLoss()


optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)