import os

from PIL import Image, ImageEnhance
from tqdm import tqdm
from ultralyticsplus import YOLO, render_result


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
    model = YOLO('kittendev/YOLOv8m-smoke-detection')
    import yolov5
    # model = yolov5.load('keremberke/yolov5s-smoke')

    # 设置模型参数
    model.overrides['conf'] = 0.4  # NMS置信度阈值
    model.overrides['iou'] = 0.3  # NMS IoU阈值
    model.overrides['agnostic_nms'] = False  # NMS类无关
    model.overrides['max_det'] = 1000  # 每张图片的最大检测数量

    # 设置使用 GPU (如果可用)
    device = 'mps'  # 'mps' 适用于苹果 M1/M2 芯片的 GPU
    model.to(device)

    # 获取输入路径下的所有图片文件
    image_files = sorted(
        [f for f in os.listdir(input_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))])

    # 遍历并处理每张图片，显示进度条
    for filename in tqdm(image_files, desc="Processing Images"):
        image_path = os.path.join(input_path, filename)

        # 压缩图片
        resized_image = resize_image(image_path)
        # resized_image_path = os.path.join('preprocess', f"resized_{filename}")
        # resized_image.save(resized_image_path)

        # 增强对比度
        # enhanced_image = enhance_contrast(resized_image)
        # enhanced_image_path = os.path.join('preprocess', f"enhanced_{filename}")
        # enhanced_image.save(enhanced_image_path)

        # 执行推理
        results = model.predict(resized_image, device=device, verbose=False, augment=True)

        # parse results
        # predictions = results.pred[0]
        # boxes = predictions[:, :4]  # x1, y1, x2, y2
        # scores = predictions[:, 4]
        # categories = predictions[:, 5]

        output_image_path = os.path.join(output_path, filename)
        # results.save(save_dir=output_path)
        #
        # 显示并保存结果
        render = render_result(model=model, image=resized_image, result=results[0])
        render.save(output_image_path)


if __name__ == "__main__":
    main()
