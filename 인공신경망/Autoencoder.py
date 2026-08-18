import torch
import torch.nn as nn
import torch.optim as optim


class AutoEncoder(nn.Module):

    def __init__(
        self,
        input_size=100
    ):
        super().__init__()


        # ====================================================
        # Encoder
        # ====================================================

        self.encoder = nn.Sequential(

            # 100 → 64
            nn.Linear(
                input_size,
                64
            ),

            # Batch Normalization
            nn.BatchNorm1d(64),

            # Activation
            nn.ReLU(),

            # Dropout
            nn.Dropout(0.2),

            # 64 → 32
            nn.Linear(
                64,
                32
            ),

            nn.ReLU(),

            # Latent Vector
            nn.Linear(
                32,
                16
            )
        )


        # ====================================================
        # Decoder
        # ====================================================

        self.decoder = nn.Sequential(

            # 16 → 32
            nn.Linear(
                16,
                32
            ),

            nn.ReLU(),

            # 32 → 64
            nn.Linear(
                32,
                64
            ),

            nn.ReLU(),

            # 원래 Feature 수로 복원
            nn.Linear(
                64,
                input_size
            )
        )


    def forward(self, x):

        # 압축
        latent = self.encoder(x)

        # 복원
        output = self.decoder(latent)

        return output


model = AutoEncoder(
    input_size=100
)


# 원본과 복원 데이터의 차이
criterion = nn.MSELoss()


optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)


# ============================================================
# Training
# ============================================================

for epoch in range(1000):

    output = model(X)

    # X 자체가 정답
    loss = criterion(
        output,
        X
    )

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()