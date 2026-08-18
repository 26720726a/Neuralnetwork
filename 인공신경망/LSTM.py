import torch
import torch.nn as nn
import torch.optim as optim


class LSTMModel(nn.Module):

    def __init__(
        self,
        input_size=5,
        hidden_size=64,
        num_classes=3
    ):
        super().__init__()

        self.lstm = nn.LSTM(

            # 각 timestep의 Feature
            input_size=input_size,

            # Hidden 크기
            hidden_size=hidden_size,

            # LSTM 두 층
            num_layers=2,

            # (batch, sequence, feature)
            batch_first=True,

            # Layer 사이 Dropout
            dropout=0.3
        )

        # 정규화
        self.norm = nn.LayerNorm(
            hidden_size
        )

        # Dropout
        self.dropout = nn.Dropout(0.3)

        # 최종 분류
        self.fc = nn.Linear(
            hidden_size,
            num_classes
        )


    def forward(self, x):

        # LSTM Forward
        output, (hidden, cell) = self.lstm(x)

        # 마지막 timestep
        x = output[:, -1, :]

        # Layer Normalization
        x = self.norm(x)

        # Dropout
        x = self.dropout(x)

        # Classification
        x = self.fc(x)

        return x


model = LSTMModel()


criterion = nn.CrossEntropyLoss()


optimizer = optim.AdamW(
    model.parameters(),
    lr=0.001,
    weight_decay=0.01
)