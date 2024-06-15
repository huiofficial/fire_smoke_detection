import os

from PIL import Image, ImageEnhance
from tqdm import tqdm


def resize_image(image_path, target_size=(640, 640)):
    with Image.open(image_path) as img:
        img_resized = img.resize(target_size)
    return img_resized


def enhance_contrast(image, factor=1.5):
    enhancer = ImageEnhance.Contrast(image)
    return enhancer.enhance(factor)


def main():
    # 设置输入和输出路径
    input_path = 'data'  # 修改为你的图片路径
    output_path = 'outputs'
    os.makedirs(output_path, exist_ok=True)
    os.makedirs('preprocess', exist_ok=True)

    # 加载模型
    # model = YOLO('kittendev/YOLOv8m-smoke-detection')
    import yolov5
    model = yolov5.load('keremberke/yolov5s-smoke')

    # 设置模型参数
    model.conf = 0.25  # NMS confidence threshold
    model.iou = 0.45  # NMS IoU threshold
    model.agnostic = False  # NMS class-agnostic
    model.multi_label = False  # NMS multiple labels per box
    model.max_det = 1000  # maximum number of detections per image

    # 设置使用 GPU (如果可用)
    device = 'mps'  # 'mps' 适用于苹果 M1/M2 芯片的 GPU
    model.to(device)

    # 获取输入路径下的所有图片文件
    image_files = sorted(
        [f for f in os.listdir(input_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))])

    # 遍历并处理每张图片，显示进度条
    for filename in tqdm(image_files, desc="Processing Images"):
        image_path = os.path.join(input_path, filename)

        # inference with test time augmentation
        results = model(image_path, augment=True)

        # 保存结果到 "results/" 文件夹，保留原始文件名
        results.save(save_dir=output_path, exist_ok=True)
        # 遍历结果并保存检测到的图片
        for i, result in enumerate(results.render()):
            result_image = Image.fromarray(result)
            result_image_path = os.path.join(output_path, f"{filename}")
            result_image.save(result_image_path)


if __name__ == "__main__":
    main()
