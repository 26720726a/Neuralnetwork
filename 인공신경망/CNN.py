import torch
import torch.nn as nn
import torch.optim as optim


class CNN(nn.Module):

    def __init__(
        self,
        in_channels=3,
        num_classes=10
    ):
        super().__init__()

        # --------------------------------
        # CNN Block 1
        # --------------------------------

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=32,
            kernel_size=3,
            padding=1
        )

        self.bn1 = nn.BatchNorm2d(32)


        # --------------------------------
        # CNN Block 2
        # --------------------------------

        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            padding=1
        )

        self.bn2 = nn.BatchNorm2d(64)


        # 활성화 함수
        self.relu = nn.ReLU()


        # 이미지 크기 절반으로 감소
        self.pool = nn.MaxPool2d(
            kernel_size=2
        )


        # Channel 단위 Dropout
        self.dropout2d = nn.Dropout2d(0.25)


        # 입력 이미지 크기에 상관없이
        # 최종 feature map을 4x4로 맞춤
        self.adaptive_pool = nn.AdaptiveAvgPool2d(
            (4, 4)
        )


        # Fully Connected
        self.fc1 = nn.Linear(
            64 * 4 * 4,
            128
        )

        self.dropout = nn.Dropout(0.5)

        self.fc2 = nn.Linear(
            128,
            num_classes
        )


    def forward(self, x):

        # CNN Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)
        # Dropout2d는 일반 Dropout과 달리 전체 채널 단위로 0을 만들 수 있어 CNN feature map에 사용할 수 있다
        x = self.dropout2d(x) 

        # CNN Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool(x)

        # Feature map 크기 통일
        x = self.adaptive_pool(x)

        # 4차원 → 2차원
        x = torch.flatten(
            x,
            start_dim=1
        )

        # Fully Connected
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)

        return x


model = CNN(
    in_channels=3,
    num_classes=10
)


criterion = nn.CrossEntropyLoss()


optimizer = optim.AdamW(
    model.parameters(),
    lr=0.001,
    weight_decay=0.01
)