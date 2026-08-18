import torch
import torch.nn as nn
import torch.optim as optim


class TransformerModel(nn.Module):

    def __init__(
        self,
        input_size=10,
        d_model=64,
        num_classes=3,
        max_len=100
    ):
        super().__init__()

        # --------------------------------
        # 입력 Feature → Transformer 차원
        # --------------------------------

        self.embedding = nn.Linear(
            input_size,
            d_model
        )


        # --------------------------------
        # 위치 정보
        #
        # 학습 가능한 Positional Embedding
        # --------------------------------

        self.position_embedding = nn.Parameter(
            torch.zeros(
                1,
                max_len,
                d_model
            )
        )


        # --------------------------------
        # Transformer Encoder Layer
        # --------------------------------

        encoder_layer = nn.TransformerEncoderLayer(

            # Transformer Feature 크기
            d_model=d_model,

            # Attention Head 수
            nhead=8,

            # Feed Forward Layer 크기
            dim_feedforward=256,

            # Dropout
            dropout=0.1,

            # 입력:
            # (batch, sequence, feature)
            batch_first=True,

            # LayerNorm 먼저 수행
            norm_first=True
        )


        # Encoder 3개 쌓기
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=3
        )


        # 마지막 정규화
        self.norm = nn.LayerNorm(
            d_model
        )


        self.dropout = nn.Dropout(0.3)


        self.fc = nn.Linear(
            d_model,
            num_classes
        )


    def forward(self, x):

        # Feature → d_model
        x = self.embedding(x)

        # 현재 sequence 길이
        seq_len = x.size(1)

        # 위치 정보 추가
        x = (
            x
            + self.position_embedding[:, :seq_len, :]
        )

        # Transformer Encoder
        x = self.transformer(x)

        # Sequence 전체 평균
        x = x.mean(dim=1)

        # Layer Normalization
        x = self.norm(x)

        # Dropout
        x = self.dropout(x)

        # 최종 분류
        x = self.fc(x)

        return x


model = TransformerModel()


criterion = nn.CrossEntropyLoss()


optimizer = optim.AdamW(
    model.parameters(),
    lr=0.0005,
    weight_decay=0.01
)