import os
import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from model import TopoUNet
from Postprocessing import FracturePostProcessor
from config import Config
import matplotlib.pyplot as plt
import argparse

def load_model(checkpoint_path, model):
    """加载训练好的模型"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

def preprocess_image(image_path, target_size=(512, 512)):
    """预处理图像"""
    # 读取图像
    image = Image.open(image_path).convert('RGB')
    
    # 调整大小
    image = image.resize(target_size, Image.BILINEAR)
    
    # 转换为张量并标准化
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    return transform(image).unsqueeze(0)

def inference(model, image_path, output_dir, device):
    """对单个图像进行推理"""
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 预处理图像
    input_tensor = preprocess_image(image_path)
    input_tensor = input_tensor.to(device)
    
    # 读取原始图像用于可视化
    original_image = cv2.imread(image_path)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    original_image = cv2.resize(original_image, (512, 512))
    
    # 模型推理
    model.eval()
    with torch.no_grad():
        seg_pred, edge_pred = model(input_tensor)
    
    # 后处理
    postprocessor = FracturePostProcessor(threshold=0.5, min_area=50)
    result = postprocessor.process(seg_pred, edge_pred, original_image)
    
    # 可视化并保存结果
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    
    # 保存二值掩码
    cv2.imwrite(
        os.path.join(output_dir, f"{base_name}_mask.png"),
        result['binary_mask']
    )
    
    # 保存骨架
    cv2.imwrite(
        os.path.join(output_dir, f"{base_name}_skeleton.png"),
        result['skeleton']
    )
    
    # 可视化结果
    vis_path = os.path.join(output_dir, f"{base_name}_visualization.png")
    postprocessor.visualize(result, save_path=vis_path)
    
    # 保存拓扑网络
    if result['graph'].number_of_nodes() > 0:
        graph_path = os.path.join(output_dir, f"{base_name}_graph.json")
        postprocessor.save_graph(result['graph'], graph_path)
    
    print(f"处理完成: {image_path}")
    print(f"结果已保存至: {output_dir}")
    
    return result

def batch_inference(model, image_dir, output_dir, device):
    """对目录中的所有图像进行批量推理"""
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有图像文件
    image_files = [f for f in os.listdir(image_dir) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    
    if not image_files:
        print(f"在 {image_dir} 中没有找到图像文件")
        return
    
    # 处理每个图像
    for image_file in image_files:
        image_path = os.path.join(image_dir, image_file)
        print(f"处理图像: {image_path}")
        inference(model, image_path, output_dir, device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="裂隙分割推理脚本")
    parser.add_argument("--checkpoint", required=True, help="模型检查点路径")
    parser.add_argument("--input", required=True, help="输入图像路径或目录")
    parser.add_argument("--output", default="results", help="输出目录")
    parser.add_argument("--device", default="cuda", help="运行设备 (cuda/cpu)")
    parser.add_argument("--batch", action="store_true", help="批量处理目录中的所有图像")
    
    args = parser.parse_args()
    
    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    print(f"使用设备: {device}")
    
    # 初始化模型
    model = TopoUNet()
    model = load_model(args.checkpoint, model)
    model = model.to(device)
    
    # 运行推理
    if args.batch:
        batch_inference(model, args.input, args.output, device)
    else:
        inference(model, args.input, args.output, device)