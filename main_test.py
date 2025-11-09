import keras
import torch
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
# --- CNN을 위한 핵심 레이어 추가 ---
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.datasets import fashion_mnist
# --- 조기 종료를 위한 콜백 추가 ---
from keras.callbacks import EarlyStopping

def main():
    print(f"Keras 백엔드: {keras.backend.backend()}")
    if torch.cuda.is_available():
        print(f"PyTorch CUDA 사용 가능 여부: {torch.cuda.is_available()}")
        print(f"현재 PyTorch 디바ICE: {torch.cuda.get_device_name(0)}")

    ## 1. Fashion MNIST 데이터셋 불러오기
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    ## 2. 데이터 스케일링
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    ## 3. (중요) 데이터 형태 변경: CNN을 위한 채널 추가
    # MLP는 (28, 28) 형태를 사용하지만, CNN은 (28, 28, 1) 형태 (가로, 세로, 채널)가 필요합니다.
    # 흑백 이미지이므로 채널은 1입니다.
    train_images = train_images.reshape((-1, 28, 28, 1))
    test_images = test_images.reshape((-1, 28, 28, 1))
    
    print(f"변경된 훈련 이미지 형태: {train_images.shape}") # (60000, 28, 28, 1)

    ## 4. 모델 구성 (MLP -> CNN으로 변경)
    model = Sequential()
    
    # C-B-P 블록 1: 32개의 3x3 필터
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)))
    model.add(BatchNormalization()) # 학습 안정화
    model.add(MaxPooling2D((2, 2))) # 14x14로 크기 축소
    
    # C-B-P 블록 2: 64개의 3x3 필터
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2))) # 7x7로 크기 축소

    # 분류기 (Classifier)
    model.add(Flatten()) # (7, 7, 64) -> 1D 벡터로 변환
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5)) # 과적합 방지를 위해 드롭아웃 비율을 0.5로 늘림
    model.add(Dense(320, activation='relu'))
    model.add(Dense(240, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    
    model.summary()

    ## 5. 모델 컴파일 및 학습
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    # (중요) 조기 종료(EarlyStopping) 설정
    # 검증(validation) 손실(val_loss)이 5번 연속 개선되지 않으면 학습을 중단합니다.
    # restore_best_weights=True : 가장 성능이 좋았던 시점의 가중치로 복원
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # (중요) validation_split=0.2 : 훈련 데이터의 20%를 검증용으로 사용
    # EarlyStopping은 이 검증 데이터의 성능(val_loss)을 모니터링합니다.
    # epochs를 50 정도로 늘려도, 조기 종료가 알아서 최적의 시점에 멈춰줍니다.
    history = model.fit(train_images, train_labels, 
                        epochs=50, 
                        batch_size=64, 
                        verbose=1,
                        validation_split=0.2,  # 검증 데이터 분할
                        callbacks=[early_stop]) # 조기 종료 콜백 적용

    ## 6. 모델 테스트
    # EarlyStopping이 가장 좋았던 가중치를 복원했으므로, 
    # 이 테스트 성능이 모델의 최고 성능입니다.
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print(f'\nTest accuracy: {test_acc:.4f}') # 소수점 4자리까지 표시

if __name__ == "__main__":
    main()