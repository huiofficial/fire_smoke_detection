import os

from ultralytics import YOLO


def validate(model, data_yaml):
    # Validate the model on the validation set specified in the data yaml file
    results = model.val(data=data_yaml)
    print(results)


def main():
    # Load a COCO-pretrained YOLOv8n model
    model = YOLO("yolov8n.pt")

    # Display model information (optional)
    model.info()

    log_dir = "runs/train/exp1"

    # Custom training parameters including data augmentation settings
    train_params = {
        'data': "zgzl_dataset.yaml",
        'epochs': 200,
        'imgsz': 640,
        'project': 'runs/train',
        'name': 'exp1',
        'patience': 100,
        'mosaic': 1.0,
        'mixup': 0.0,
        'hsv_h': 0.015,
        'hsv_s': 0.7,
        'hsv_v': 0.4,
        'degrees': 0.0,
        'translate': 0.1,
        'scale': 0.5,
        'shear': 0.0,
        'perspective': 0.0,
        'flipud': 0.0,
        'fliplr': 0.5,
    }

    # Train the model with custom augmentation parameters
    results = model.train(**train_params, device="mps", amp=True, workers=8, exist_ok=True, single_cls=True)

    # Validate the model on the zgzl_dataset validation set
    validate(model, "zgzl_dataset.yaml")

    # Start TensorBoard
    os.system(f"tensorboard --logdir {log_dir} --host 0.0.0.0 --port 6006")


if __name__ == '__main__':
    main()
