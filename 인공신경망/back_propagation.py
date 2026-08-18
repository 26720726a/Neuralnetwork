import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# -------------------------
# 1. 순전파
# -------------------------
def forward(x, y_true, w1, b1, w2, b2):

    z1 = np.dot(w1, x) + b1
    h = sigmoid(z1)

    z2 = np.dot(w2, h) + b2
    y_pred = sigmoid(z2)

    loss = np.mean((y_pred - y_true) ** 2)

    return y_pred, h, loss


# -------------------------
# 2. 역전파
# -------------------------
def backward(x, y_true, y_pred, h, w2):

    # 출력층
    delta2 = 2 * (y_pred - y_true) * y_pred * (1 - y_pred)

    dw2 = np.outer(delta2, h)
    db2 = delta2

    # 은닉층
    delta1 = np.dot(w2.T, delta2) * h * (1 - h)

    dw1 = np.outer(delta1, x)
    db1 = delta1

    return dw1, db1, dw2, db2


# -------------------------
# 3. 가중치 업데이트
# -------------------------
def update(w1, b1, w2, b2,
           dw1, db1, dw2, db2,
           lr):

    w1 = w1 - lr * dw1
    b1 = b1 - lr * db1

    w2 = w2 - lr * dw2
    b2 = b2 - lr * db2

    return w1, b1, w2, b2


def main():

    x = np.array([1.0, 0.0])
    y_true = np.array([0.0])

    w1 = np.array([
        [0.1, 0.2],
        [0.3, 0.4]
    ])

    b1 = np.array([0.1, 0.2])

    w2 = np.array([
        [0.5, 0.6]
    ])

    b2 = np.array([0.3])

    lr = 0.5

    # 1. 순전파
    y_pred, h, loss = forward(
        x, y_true,
        w1, b1,
        w2, b2
    )

    print("학습 전 loss:", loss)

    # 2. 역전파
    dw1, db1, dw2, db2 = backward(
        x,
        y_true,
        y_pred,
        h,
        w2
    )

    # 3. 가중치 업데이트
    w1, b1, w2, b2 = update(
        w1, b1, w2, b2,
        dw1, db1, dw2, db2,
        lr
    )

    # 4. 새로운 가중치로 다시 순전파
    y_pred, h, loss = forward(
        x, y_true,
        w1, b1,
        w2, b2
    )

    print("학습 후 loss:", loss)


main()