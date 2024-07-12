import os
import shutil
import time
from functools import wraps

from PIL import Image
from torchvision import transforms
from ultralytics import YOLO
from ultralyticsplus import render_result


def clear_or_create_dir(directory):
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory, exist_ok=True)


def resize_image(img_path, scale=0.25):
    image = Image.open(img_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((int(image.height * scale), int(image.width * scale)))
    ])
    resized_img = transform(image)
    return resized_img


def predict_and_save(model_path, input_dir, output_dir):
    # 加载训练好的模型
    model = YOLO(model_path)

    # 清空或创建输出目录
    clear_or_create_dir(output_dir)

    # 获取并排序输入目录中的所有图片文件名
    img_files = sorted([f for f in os.listdir(input_dir) if f.endswith(('.JPG', '.jpg', '.png', '.bmp'))])

    # 使用 tqdm 显示预测进度
    for img_name in img_files:
        img_path = os.path.join(input_dir, img_name)

        # 加载并压缩图像
        resized_img = resize_image(img_path, scale=0.25)

        # 对压缩图像进行预测
        results = model.predict(source=resized_img, device="mps")

        # 显示并保存结果
        render = render_result(model=model, image=resized_img, result=results[0])
        output_image_path = os.path.join(output_dir, img_name)
        render.save(output_image_path)


def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Total execution time: {elapsed_time:.2f} seconds")
        return result

    return wrapper


@timer
def main():
    # 训练好的模型路径
    trained_model_path = "runs/train/exp1/weights/best.pt"

    # 待预测的图片目录
    input_dir = "data"

    # 保存预测结果的目录
    output_dir = "outputs"

    # 执行预测并保存结果
    predict_and_save(trained_model_path, input_dir, output_dir)


if __name__ == '__main__':
    main()
