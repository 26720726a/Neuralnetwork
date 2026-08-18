MLP

Linear
 ↓
BatchNorm1d
 ↓
ReLU
 ↓
Dropout


CNN

Conv2d
 ↓
BatchNorm2d
 ↓
ReLU
 ↓
Pooling
 ↓
Dropout / Dropout2d


RNN / LSTM / GRU

RNN 계층
 ↓
마지막 timestep
 ↓
Dropout
 ↓
Linear


Transformer

Embedding
 ↓
Transformer Encoder
 ↓
Pooling
 ↓
Dropout
 ↓
Linear

일반 데이터 
nn.Linear()
nn.BatchNorm1d()
nn.ReLU()
nn.Dropout()

이미지
nn.Conv2d()
nn.BatchNorm2d()
nn.ReLU()
nn.MaxPool2d()
torch.flatten()
nn.Linear()

시계열
nn.RNN()
nn.LSTM()
nn.GRU()
x = output[:, -1, :]

transformer
nn.TransformerEncoderLayer()
nn.TransformerEncoder()

공통학습
output = model(X)

loss = criterion(output, y)

optimizer.zero_grad()

loss.backward()

optimizer.step()


| 데이터      | 추천 모델       | 주요 Layer                  | BatchNorm     |
| -------- | ----------- | ------------------------- | ------------- |
| 일반 숫자    | MLP/DNN     | `Linear`                  | `BatchNorm1d` |
| 이미지      | CNN         | `Conv2d`, `Pool`          | `BatchNorm2d` |
| 시계열      | RNN         | `RNN`                     | 보통 별도 처리      |
| 긴 시계열    | LSTM        | `LSTM`                    | 보통 별도 처리      |
| 시계열      | GRU         | `GRU`                     | 보통 별도 처리      |
| Sequence | Transformer | `TransformerEncoder`      | 내부 정규화 구조 활용  |
| 압축/복원    | Autoencoder | `Linear/Conv`             | 선택            |
| 생성       | GAN         | Generator + Discriminator | 모델별 선택        |

| 모델          | 입력 shape    | 입력이 바뀌면 수정할 곳             |
| ----------- | ----------- | ------------------------- |
| DNN         | `(B,F)`     | `Linear(F, ...)`          |
| CNN         | `(B,C,H,W)` | 첫 `Conv2d(in_channels=C)` |
| RNN         | `(B,S,F)`   | `RNN(input_size=F)`       |
| LSTM        | `(B,S,F)`   | `LSTM(input_size=F)`      |
| GRU         | `(B,S,F)`   | `GRU(input_size=F)`       |
| Transformer | `(B,S,F)`   | `Linear(F,d_model)`       |
| Autoencoder | `(B,F)`     | 첫/마지막 `Linear`의 `F`       |



| 문제 종류 |   마지막 출력 | 마지막 활성화 | Loss                | 최종 예측             |
| ----- | -------: | ------- | ------------------- | ----------------- |
| 이진분류  |        1 | 없음      | `BCEWithLogitsLoss` | `sigmoid → >=0.5` |
| 다중분류  |    클래스 수 | 없음      | `CrossEntropyLoss`  | `argmax`          |
| 다중라벨  |    클래스 수 | 없음      | `BCEWithLogitsLoss` | `sigmoid → >=0.5` |
| 회귀    | 필요한 숫자 수 | 보통 없음   | `MSELoss`           | 출력 그대로            |
