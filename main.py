def main():
    # Python 스크립트 또는 REPL에서 실행
    import keras
    import torch

    print(f"Keras 백엔드: {keras.backend.backend()}")
    print(f"PyTorch CUDA 사용 가능 여부: {torch.cuda.is_available()}")
    print(f"현재 PyTorch 디바이스: {torch.cuda.get_device_name(0)}")


if __name__ == "__main__":
    main()
