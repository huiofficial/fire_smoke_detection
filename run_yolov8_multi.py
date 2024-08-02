import os
import shutil
import time
from functools import wraps

import torch
from PIL import Image
from torchvision import transforms
from torchvision.ops import nms
from ultralytics import YOLO
from ultralyticsplus import render_result


def clear_or_create_dir(directory):
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory, exist_ok=True)


def resize_image(image, scale=1.0):
    transform = transforms.Compose([
        transforms.Resize((int(image.height * scale), int(image.width * scale)))
    ])
    resized_img = transform(image)
    return resized_img


def split_image(image, rows=3, cols=3):
    width, height = image.size
    grid_w, grid_h = width // cols, height // rows
    grid_images = []

    for row in range(rows):
        for col in range(cols):
            left, upper, right, lower = col * grid_w, row * grid_h, (col + 1) * grid_w, (row + 1) * grid_h
            grid_images.append(image.crop((left, upper, right, lower)))

    return grid_images, grid_w, grid_h


def multi_scale_predict(model, img, scales=[0.5, 1.0, 1.5], device="mps"):
    all_boxes = []
    all_scores = []
    all_labels = []

    for scale in scales:
        resized_img = resize_image(img, scale=scale)
        results = model.predict(source=resized_img, device=device)

        boxes = results[0].boxes.xyxy
        scores = results[0].boxes.conf
        labels = results[0].boxes.cls

        all_boxes.append(boxes)
        all_scores.append(scores)
        all_labels.append(labels)

    all_boxes = torch.cat(all_boxes)
    all_scores = torch.cat(all_scores)
    all_labels = torch.cat(all_labels)

    # Apply NMS to merge results
    keep_indices = nms(all_boxes, all_scores, iou_threshold=0.5)
    final_boxes = all_boxes[keep_indices]
    final_scores = all_scores[keep_indices]
    final_labels = all_labels[keep_indices]

    return final_boxes, final_scores, final_labels


def predict_and_save(model_path, input_dir, output_dir, scales=[0.5, 1.0, 1.5], device="mps"):
    # 加载训练好的模型
    model = YOLO(model_path)

    # 清空或创建输出目录
    clear_or_create_dir(output_dir)

    # 获取并排序输入目录中的所有图片文件名
    img_files = sorted([f for f in os.listdir(input_dir) if f.endswith(('.JPG', '.jpg', '.png', '.bmp'))])

    # 使用 tqdm 显示预测进度
    for img_name in img_files:
        img_path = os.path.join(input_dir, img_name)
        image = Image.open(img_path).convert("RGB")

        # 将图像分割为3x3网格
        grid_images, grid_w, grid_h = split_image(image, rows=3, cols=3)

        all_boxes = []
        all_scores = []
        all_labels = []

        for i, grid_img in enumerate(grid_images):
            boxes, scores, labels = multi_scale_predict(model, grid_img, scales=scales, device=device)

            # 调整坐标到原图位置
            row, col = divmod(i, 3)
            boxes[:, [0, 2]] += col * grid_w
            boxes[:, [1, 3]] += row * grid_h

            all_boxes.append(boxes)
            all_scores.append(scores)
            all_labels.append(labels)

        all_boxes = torch.cat(all_boxes)
        all_scores = torch.cat(all_scores)
        all_labels = torch.cat(all_labels)

        # Apply NMS to merge results
        keep_indices = nms(all_boxes, all_scores, iou_threshold=0.5)
        final_boxes = all_boxes[keep_indices]
        final_scores = all_scores[keep_indices]
        final_labels = all_labels[keep_indices]

        # 显示并保存结果
        result = {"boxes": final_boxes, "scores": final_scores, "labels": final_labels}
        render = render_result(model=model, image=image, result=result)
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
