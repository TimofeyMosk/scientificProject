from multiprocessing import freeze_support
from ultralytics import YOLO


def main():
    freeze_support()
    model = YOLO("yolov8m.pt")

    results = model.train(
        data='dataset/yolov8.yaml',
        imgsz=320,  # 640
        epochs=40,
        batch=4,
        name='FINALyolov8m_custom')


if __name__ == '__main__':
    main()
