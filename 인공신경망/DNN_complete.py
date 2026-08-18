import torch
import torch.nn as nn
import torch.optim as optim

# 같은 폴더의 train_eval.py 에서 불러옴
from train_eval import train_model, evaluate


# =========================================================
# 1. 데이터 준비
# =========================================================
# ※ 실제 시험에서는 이 블록만 주어진 데이터로 교체하면 됨
#    (예: pd.read_csv 로 읽어 tensor 변환, 또는 제공된 X, y 사용)
#   pd.read_csv("MSFT.csv")
#    - 분류이므로 y 는 원-핫이 아니라 "클래스 인덱스(long)" 여야 함
#
# 여기서는 데모용으로 10차원 공간에 3개 클래스 덩어리(blob)를 생성
torch.manual_seed(0)

input_size = 10
num_classes = 3
n_per_class = 300

centers = torch.randn(num_classes, input_size) * 4.0   # 클래스별 중심

X_list, y_list = [], []
for c in range(num_classes):
    X_list.append(torch.randn(n_per_class, input_size) + centers[c])
    y_list.append(torch.full((n_per_class,), c, dtype=torch.long))

X = torch.cat(X_list, dim=0)     # (900, 10)
y = torch.cat(y_list, dim=0)     # (900,)

# 순서 섞기
perm = torch.randperm(X.shape[0])
X, y = X[perm], y[perm]

# test set 을 따로 20% 떼어둠 (나머지는 train_model 안에서 train/val 분할)
n_test = int(len(X) * 0.2)
X_test, y_test = X[:n_test], y[:n_test]
X_trainval, y_trainval = X[n_test:], y[n_test:]


# =========================================================
# 2. 모델 정의 (DNN 템플릿 그대로)
# =========================================================
class DNN(nn.Module):

    def __init__(self, input_size=10, num_classes=3):
        super().__init__()

        self.fc1 = nn.Linear(input_size, 64)
        self.bn1 = nn.BatchNorm1d(64)

        self.fc2 = nn.Linear(64, 32)
        self.bn2 = nn.BatchNorm1d(32)

        self.fc3 = nn.Linear(32, num_classes)

        self.relu = nn.ReLU()
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

        # 출력층 (Softmax 없음 -> CrossEntropyLoss 가 내부 처리)
        x = self.fc3(x)
        return x


# =========================================================
# 3. 모델 / 손실 / 옵티마이저
# =========================================================
model = DNN(input_size=input_size, num_classes=num_classes)

criterion = nn.CrossEntropyLoss()              # 다중분류

optimizer = optim.AdamW(
    model.parameters(),
    lr=0.001,
    weight_decay=0.01
)


# =========================================================
# 4. 학습
#    (train/val 분할 + Early Stopping + best 저장 + Learning Curve)
# =========================================================
history, model = train_model(
    model, criterion, optimizer,
    X_trainval, y_trainval,        # 전체 -> 내부에서 4:1 분할
    task="classification",
    val_ratio=0.2,                 # Train : Val = 4 : 1
    epochs=1000,
    patience=20,                   # 검증손실 20에폭 개선 없으면 종료
    batch_size=64,
    save_path="best_dnn.pth",      # 가장 좋은 모델 저장 경로
)


# =========================================================
# 5. 평가 (test set 정확도)
# =========================================================
acc = evaluate(model, X_test, y_test, task="classification")
print("Test accuracy:", round(acc, 4))