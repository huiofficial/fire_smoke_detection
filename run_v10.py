import os

from ultralytics import YOLO


def main():
    # 设置输入和输出路径
    input_path = 'data'  # 修改为你的图片路径
    output_path = 'outputs'
    os.makedirs(output_path, exist_ok=True)
    os.makedirs('preprocess', exist_ok=True)

    # 获取输入路径下的所有图片文件
    image_files = sorted(
        [f for f in os.listdir(input_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))])

    # Load a model
    model = YOLO("jameslahm/yolov10x").to("mps")  # pretrained YOLOv8n model

    # Run batched inference on a list of images
    results = model(image_files)  # return a list of Results objects

    # Process results list
    for i, result in enumerate(results):
        result.save(filename="outputs/result_i.jpg")  # save to disk


if __name__ == "__main__":
    main()
