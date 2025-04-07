import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2
from Postprocessing import FracturePostProcessor

class CrackDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None):
        """
        Args:
            image_dir (str): 原始图像目录路径
            label_dir (str): 标签图像目录路径
            transform (callable, optional): 数据增强转换
        """
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.postprocessor = FracturePostProcessor(threshold=0.5, min_area=10)  # 初始化后处理器
        # 获取所有图像文件名
        self.images = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
        
        # 获取所有标签文件名以创建映射
        self.label_files = sorted([f for f in os.listdir(label_dir) if f.endswith('.png')])
        
        # 创建映射字典: 图像文件名 -> 标签文件名
        self.img_to_label_map = self._create_filename_mapping()
        
        print(f"加载了 {len(self.images)} 张图像")
        print(f"加载了 {len(self.label_files)} 张标签")
        print(f"成功匹配了 {len(self.img_to_label_map)} 对图像-标签对")
        
    def _create_filename_mapping(self):
        """创建图像文件名到标签文件名的映射"""
        mapping = {}
        
        # 从图像文件名中提取数字部分
        for img_file in self.images:
            # 提取数字部分 (假设文件名格式为 "数字.png")
            img_num = int(os.path.splitext(img_file)[0])
            
            # 查找对应的标签文件 (格式为 "000数字.png")
            expected_label_format = f"{img_num:06d}.png"  # 格式化为6位数，前面补零
            
            # 如果存在精确匹配的文件
            if expected_label_format in self.label_files:
                mapping[img_file] = expected_label_format
            else:
                # 尝试其他可能的格式 (如 "000数字.png")
                for label_file in self.label_files:
                    label_num = int(os.path.splitext(label_file)[0].lstrip('0') or '0')
                    if label_num == img_num:
                        mapping[img_file] = label_file
                        break
        
        return mapping
        
    def __len__(self):
        return len(self.img_to_label_map)
    
    def __getitem__(self, idx):
        # 获取图像文件名及其对应的标签文件名
        img_name = self.images[idx]
        if img_name not in self.img_to_label_map:
            raise ValueError(f"找不到图像 '{img_name}' 对应的标签文件")
            
        label_name = self.img_to_label_map[img_name]

        # 构建完整路径
        img_path = os.path.join(self.image_dir, img_name)
        label_path = os.path.join(self.label_dir, label_name)
        
        # 打印调试信息 (首次加载时)
        if idx == 0:
            print(f"图像路径: {img_path}")
            print(f"标签路径: {label_path}")
        
        # 读取原始图像
        image = Image.open(img_path).convert('RGB')
        
        # 读取标签图像
        label = Image.open(label_path).convert('RGB')  # 确保标签是 RGB 格式
        
        # 首先调整图像和标签到相同大小
        target_size = (512, 512)
        image = image.resize(target_size, Image.BILINEAR)
        label = label.resize(target_size, Image.NEAREST)  # 使用最近邻插值保持标签的离散性
        
        # 将标签转换为numpy数组
        label_np = np.array(label)
        
        # 提取所有非黑色部分
        binary_mask = np.any(label_np > 0, axis=-1).astype(np.float32)
        
        # 打印调试信息
        if idx == 0:
            print(f"二值掩码中非零像素数: {np.sum(binary_mask)}")
        
        # 使用后处理器处理二值掩码
        topo_result = self.postprocessor.process(binary_mask)
        skeleton = topo_result['skeleton']  # 骨架图
        graph = topo_result['graph']  # 拓扑图

        # 转换为张量
        skeleton_tensor = torch.from_numpy(skeleton).float() / 255.0

        # 数据转换
        if self.transform:
            image = self.transform(image)
        
        # 转换为张量
        label_tensor = torch.from_numpy(binary_mask)
        
        # 使用Canny边缘检测生成边缘图
        edge_map = cv2.Canny((binary_mask * 255).astype(np.uint8), threshold1=50, threshold2=150)
        edge_tensor = torch.from_numpy(edge_map).float() / 255.0
        
        return image, label_tensor.unsqueeze(0), edge_tensor.unsqueeze(0), skeleton_tensor.unsqueeze(0)
    
    def generate_edge_map(self, mask):
        """生成边缘图"""
        kernel = np.ones((3,3), np.uint8)
        dilated = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)
        eroded = cv2.erode(mask.astype(np.uint8), kernel, iterations=1)
        edge = dilated - eroded
        return edge.astype(np.float32)

def get_dataloader(
    train_img_dir,
    train_label_dir,
    val_img_dir,
    val_label_dir,
    test_img_dir,
    test_label_dir,
    batch_size=4
):
    """
    创建训练、验证和测试数据加载器

    Args:
        train_img_dir (str): 训练图像目录
        train_label_dir (str): 训练标签目录
        val_img_dir (str): 验证图像目录
        val_label_dir (str): 验证标签目录
        test_img_dir (str): 测试图像目录
        test_label_dir (str): 测试标签目录
        batch_size (int): 批次大小

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # 定义数据转换
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # 创建训练集数据加载器
    train_dataset = CrackDataset(
        image_dir=train_img_dir,
        label_dir=train_label_dir,
        transform=transform
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    # 创建验证集数据加载器
    val_dataset = CrackDataset(
        image_dir=val_img_dir,
        label_dir=val_label_dir,
        transform=transform
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    # 创建测试集数据加载器
    test_dataset = CrackDataset(
        image_dir=test_img_dir,
        label_dir=test_label_dir,
        transform=transform
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader