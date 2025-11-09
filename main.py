def main():
    # Python 스크립트 또는 REPL에서 실행
    # powershell "Set-Item -Path Env:KERAS_BACKEND -Value "torch""
    import keras
    import torch

    print(f"Keras 백엔드: {keras.backend.backend()}")
    print(f"PyTorch CUDA 사용 가능 여부: {torch.cuda.is_available()}")
    print(f"현재 PyTorch 디바이스: {torch.cuda.get_device_name(0)}")

    ## 필요한 패키지 로드
    from keras.models import Sequential
    from keras.layers import Dense, Flatten
    from keras.datasets import fashion_mnist
    import numpy as np
    import matplotlib.pyplot as plt

    ## Fashion MNIST 데이터셋 불러오기
    #fashion_mnist = keras.datasets.fashion_mnist # 중복으로 인해 실행 오류로 주석처리
    (train_images,train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    ## 데이터 시각화
    plt.figure()
    plt.imshow(train_images[0])
    plt.colorbar()
    plt.grid(False)

    plt.figure(figsize=(10,10))
    for i in range(25):
        plt.subplot(5, 5, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(train_images[i], cmap=plt.cm.binary)
        plt.xlabel(class_names[train_labels[i]])
    plt.show()

    # 데이터 스케일링
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    # 모델 구성
    model = Sequential()
    model.add(Flatten(input_shape=(28, 28)))
    model.add(Dense(240, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    model.summary()

    # 모델 컴파일 및 학습
    model.compile(optimizer='adam',
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])

    model.fit(train_images, train_labels, epochs=10, batch_size=64, verbose=1)

# 모델 테스트
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print('Test accuracy:', test_acc)

    predictions = model.predict(test_images)
    print(predictions[0])
    print(np.argmax(predictions[0]))


if __name__ == "__main__":
    main()
