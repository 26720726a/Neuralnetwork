import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input, Conv2D, BatchNormalization, MaxPooling2D, Dropout,
    Flatten, Dense, RandomFlip, RandomTranslation,
)
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

class_names = ["airplane", "automobile", "bird", "cat", "deer",
               "dog", "frog", "horse", "ship", "truck"]


# =========================================================
# 1.3 모델 설계 (입력 32x32x3, 출력 10, 파라미터 500만개 이하)
# =========================================================
def build_model():
    model = Sequential([
        Input(shape=(32, 32, 3)),

        # 가벼운 데이터 증강 (학습 때만 동작, 평가 때는 통과)
        RandomFlip("horizontal"),
        RandomTranslation(0.1, 0.1),

        # Block 1
        Conv2D(32, 3, padding="same", activation="relu"),
        BatchNormalization(),
        Conv2D(32, 3, padding="same", activation="relu"),
        BatchNormalization(),
        MaxPooling2D(2),
        Dropout(0.25),

        # Block 2
        Conv2D(64, 3, padding="same", activation="relu"),
        BatchNormalization(),
        Conv2D(64, 3, padding="same", activation="relu"),
        BatchNormalization(),
        MaxPooling2D(2),
        Dropout(0.3),

        # Block 3
        Conv2D(128, 3, padding="same", activation="relu"),
        BatchNormalization(),
        Conv2D(128, 3, padding="same", activation="relu"),
        BatchNormalization(),
        MaxPooling2D(2),
        Dropout(0.4),

        # 분류기
        Flatten(),
        Dense(128, activation="relu"),
        BatchNormalization(),
        Dropout(0.5),
        Dense(10, activation="softmax"),   # 10개 클래스 확률
    ])
    return model


if __name__ == "__main__":
    # =====================================================
    # 1.1 데이터 로드 및 전처리
    # =====================================================
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # 원본 정수 레이블은 1.2 출력에 사용하려고 따로 보관
    y_train_int = y_train.flatten()

    # 픽셀값 0~1 정규화
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # 레이블 원-핫 인코딩
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    print("x_train:", x_train.shape, "y_train:", y_train.shape)

    # =====================================================
    # 1.2 훈련셋 출력 (맨 앞 10개 이미지 + 레이블)
    # =====================================================
    plt.figure(figsize=(15, 3))
    for i in range(10):
        plt.subplot(1, 10, i + 1)
        plt.imshow(x_train[i])
        plt.title(class_names[y_train_int[i]])
        plt.axis("off")
    plt.tight_layout()
    plt.show()

    # =====================================================
    # 1.3 모델 생성 + summary
    # =====================================================
    model = build_model()
    model.summary()

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    # =====================================================
    # 1.4 모델 훈련
    # =====================================================
    # Early Stopping: 검증 손실이 10 에포크 개선 안 되면 종료,
    #                 restore_best_weights 로 가장 좋은 가중치 복원
    early_stop = EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True
    )
    # 가장 성능이 좋은 모델을 파일로 저장
    checkpoint = ModelCheckpoint(
        "best_cifar_model.keras", monitor="val_loss", save_best_only=True
    )

    history = model.fit(
        x_train, y_train,
        validation_split=0.2,     # Train : Validation = 4 : 1
        batch_size=64,
        epochs=100,
        callbacks=[early_stop, checkpoint],
        verbose=1,
    )

    # Learning Curve
    plt.figure()
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Learning Curve")
    plt.legend()
    plt.show()

    # =====================================================
    # 1.5 모델 평가 (Test set, accuracy)
    # =====================================================
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print("Test accuracy:", test_acc)