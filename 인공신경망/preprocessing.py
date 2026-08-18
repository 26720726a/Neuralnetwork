import torch

from torch.utils.data import (
    TensorDataset,
    DataLoader,
    random_split
)

from torch.nn.utils.rnn import pad_sequence


# ============================================================
# 1. 일반 데이터를 Tensor로 변환
# ============================================================

def to_tensor(
    X,
    y=None,
    classification=True
):
    """
    X:
        입력 데이터

    y:
        정답 데이터

    classification:
        True  -> 다중분류
        False -> 회귀 / 이진분류 등에 맞게 직접 조절 가능
    """

    # 입력 데이터는 기본적으로 float32
    X = torch.tensor(
        X,
        dtype=torch.float32
    )

    if y is None:
        return X

    # 다중 클래스 분류
    # CrossEntropyLoss 사용 시
    if classification:

        y = torch.tensor(
            y,
            dtype=torch.long
        )

    # 회귀 / BCE 등
    else:

        y = torch.tensor(
            y,
            dtype=torch.float32
        )

    return X, y


# ============================================================
# 2. Standardization
#
# 평균 = 0
# 표준편차 = 1
#
# DNN / RNN / LSTM / GRU 등에 사용 가능
# ============================================================

class StandardScaler:

    def __init__(self):

        self.mean = None
        self.std = None


    def fit(self, X):

        # Feature별 평균
        self.mean = X.mean(
            dim=0,
            keepdim=True
        )

        # Feature별 표준편차
        self.std = X.std(
            dim=0,
            keepdim=True
        )

        # 0으로 나누는 문제 방지
        self.std[self.std == 0] = 1.0

        return self


    def transform(self, X):

        return (
            X - self.mean
        ) / self.std


    def fit_transform(self, X):

        self.fit(X)

        return self.transform(X)


# ============================================================
# 3. Min-Max Normalization
#
# 데이터를 0 ~ 1 범위로 변환
# ============================================================

class MinMaxScaler:

    def __init__(self):

        self.min = None
        self.max = None


    def fit(self, X):

        self.min = X.min(
            dim=0,
            keepdim=True
        ).values

        self.max = X.max(
            dim=0,
            keepdim=True
        ).values

        return self


    def transform(self, X):

        diff = self.max - self.min

        # 0으로 나누는 문제 방지
        diff[diff == 0] = 1.0

        return (
            X - self.min
        ) / diff


    def fit_transform(self, X):

        self.fit(X)

        return self.transform(X)


# ============================================================
# 4. Train / Validation Dataset 분리
# ============================================================

def split_dataset(
    X,
    y,
    train_ratio=0.8
):

    # X와 y 묶기
    dataset = TensorDataset(
        X,
        y
    )

    total_size = len(dataset)

    train_size = int(
        total_size * train_ratio
    )

    val_size = (
        total_size
        - train_size
    )

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size]
    )

    return (
        train_dataset,
        val_dataset
    )


# ============================================================
# 5. DataLoader 생성
# ============================================================

def make_dataloader(
    dataset,
    batch_size=32,
    shuffle=True
):

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle
    )

    return loader


# ============================================================
# 6. Train / Validation Loader 한번에 생성
# ============================================================

def make_train_val_loader(
    X,
    y,
    batch_size=32,
    train_ratio=0.8
):

    train_dataset, val_dataset = split_dataset(
        X,
        y,
        train_ratio=train_ratio
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return (
        train_loader,
        val_loader
    )


# ============================================================
# 7. Sequence Padding
#
# RNN / LSTM / GRU / Transformer
#
# Sequence 길이가 서로 다를 때 사용
# ============================================================

def sequence_padding(
    sequences,
    padding_value=0.0
):

    """
    예:

    데이터 1 길이 = 3
    데이터 2 길이 = 5
    데이터 3 길이 = 4

    ↓

    가장 긴 길이 5에 맞춰 Padding
    """

    sequences = [
        torch.tensor(
            seq,
            dtype=torch.float32
        )
        for seq in sequences
    ]

    padded = pad_sequence(
        sequences,

        # 결과:
        # (batch, sequence, feature)
        batch_first=True,

        # 부족한 부분을 0으로 채움
        padding_value=padding_value
    )

    return padded


# ============================================================
# 8. 이미지 값 0 ~ 1 Normalize
#
# 이미지 픽셀 값이 0 ~ 255인 경우
# ============================================================

def normalize_image(X):

    X = X.float()

    # 0~255라면 0~1로 변경
    if X.max() > 1:

        X = X / 255.0

    return X


# ============================================================
# 9. CNN 입력 Shape 변경
#
# (Batch, Height, Width)
#       ↓
# (Batch, Channel, Height, Width)
#
# 흑백 이미지에 사용
# ============================================================

def add_channel_dimension(X):

    # 현재:
    # (B, H, W)

    if X.dim() == 3:

        X = X.unsqueeze(1)

    # 결과:
    # (B, 1, H, W)

    return X


# ============================================================
# 10. Binary Classification Label Shape
#
# (batch,)
#    ↓
# (batch,1)
#
# BCEWithLogitsLoss 사용 시 유용
# ============================================================

def binary_label_shape(y):

    y = y.float()

    if y.dim() == 1:

        y = y.unsqueeze(1)

    return y