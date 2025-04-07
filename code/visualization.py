import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import scipy.ndimage as ndimage
from torchvision.utils import make_grid

# 设置matplotlib字体
plt.rcParams['font.sans-serif'] = ['SimSun']  # 使用宋体
plt.rcParams['axes.unicode_minus'] = False     # 解决负号显示问题

def denormalize_image(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    反归一化图像数据
    
    Args:
        image: [C,H,W] 或 [B,C,H,W] 的张量
        mean: 归一化时使用的均值
        std: 归一化时使用的标准差
    """
    # 确保数据在CPU上
    if isinstance(image, torch.Tensor):
        image = image.cpu()
        
    # 根据维度选择正确的处理方式
    if len(image.shape) == 4:  # [B,C,H,W]
        mean = torch.tensor(mean).view(1, 3, 1, 1)
        std = torch.tensor(std).view(1, 3, 1, 1)
    else:  # [C,H,W]
        mean = torch.tensor(mean).view(3, 1, 1)
        std = torch.tensor(std).view(3, 1, 1)
    
    # 反归一化
    image = image * std + mean
    
    # 转换为numpy数组并裁剪到[0,1]范围
    if isinstance(image, torch.Tensor):
        image = image.numpy()
    
    return np.clip(image, 0, 1)

def visualize_feature_heatmap(features, save_path=None, smooth_sigma=2.0):
    """
    可视化特征热力图
    
    Args:
        features: [C,H,W] 特征图张量
        save_path: 保存路径
        smooth_sigma: 高斯平滑的sigma值
    """
    # 确保输入是tensor并且在CPU上
    if isinstance(features, torch.Tensor):
        features = features.detach().cpu()
    
    # 计算特征响应图
    response_map = torch.mean(torch.abs(features), dim=0).numpy()
    
    # 应用高斯平滑
    response_map = ndimage.gaussian_filter(response_map, sigma=smooth_sigma)
    
    # 归一化到[0,1]
    response_map = (response_map - response_map.min()) / (response_map.max() - response_map.min() + 1e-8)
    
    # 创建热力图
    plt.figure(figsize=(10, 10))
    plt.imshow(response_map, cmap='hot')
    plt.colorbar(label='特征响应强度')
    plt.title('特征激活图')
    plt.axis('off')
    
    if save_path:
        # 确保保存目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()

def visualize_vector_field(velocity_field, original_image=None, save_path=None, smooth_sigma=2.0, alpha=0.7):
    """
    将矢量场可视化并可选择叠加在原始图像上
    
    Args:
        velocity_field: [2, H, W] 速度场张量 (x方向和y方向)
        original_image: 原始图像张量 [3,H,W]（可选）
        save_path: 保存路径
        smooth_sigma: 高斯平滑的sigma值
        alpha: 叠加透明度
    """
    import numpy as np
    from scipy import ndimage
    
    # 转换为numpy数组
    if isinstance(velocity_field, torch.Tensor):
        vx = velocity_field[0].cpu().numpy()
        vy = velocity_field[1].cpu().numpy()
    else:
        vx = velocity_field[0]
        vy = velocity_field[1]
    
    # 应用高斯平滑
    vx = ndimage.gaussian_filter(vx, sigma=smooth_sigma)
    vy = ndimage.gaussian_filter(vy, sigma=smooth_sigma)
    
    # 创建网格点
    h, w = vx.shape
    y, x = np.mgrid[0:h:1, 0:w:1]
    
    # 计算速度场大小
    magnitude = np.sqrt(vx**2 + vy**2)
    
    # 创建图像
    plt.figure(figsize=(12, 8))
    
    # 显示原始图像（如果提供）
    if original_image is not None:
        if isinstance(original_image, torch.Tensor):
            img = original_image.permute(1, 2, 0).cpu().numpy()
            # 归一化图像
            img = (img - img.min()) / (img.max() - img.min())
            plt.imshow(img)
    
    # 绘制流线图 - streamplot不支持alpha参数，需要先创建流线图，然后设置其alpha值
    stream = plt.streamplot(x, y, vx, vy, 
                  density=2,  # 增加流线密度
                  color=magnitude,
                  cmap='viridis',
                  linewidth=1.5,
                  arrowsize=1.5,
                  integration_direction='forward')
    
    # 设置流线的alpha值
    stream.lines.set_alpha(alpha)
    
    # 添加半透明的热力图背景
    plt.imshow(magnitude, 
              extent=[0, w, 0, h],
              origin='lower',
              cmap='hot',
              alpha=0.3)  # 降低透明度以便看清流线
    
    plt.colorbar(label='速度大小')
    plt.title('裂纹扩展矢量场')
    plt.xlim(0, w)
    plt.ylim(0, h)
    
    if save_path:
        # 确保保存目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()


def visualize_probability_map(prob_map, original_image=None, save_path=None, title='概率分布', 
                            cmap='jet', smooth_sigma=1.0, alpha=0.6):
    """
    可视化概率图（通用函数）
    
    Args:
        prob_map: [H,W] 概率图张量
        original_image: [3,H,W] 原始图像张量（可选）
        save_path: 保存路径
        title: 图像标题
        cmap: 颜色映射
        smooth_sigma: 平滑系数
        alpha: 透明度
    """
    from scipy import ndimage
    
    # 转换为numpy数组
    if isinstance(prob_map, torch.Tensor):
        prob_map = prob_map.cpu().numpy()
    
    # 平滑概率图
    prob_map_smooth = ndimage.gaussian_filter(prob_map, sigma=smooth_sigma)
    
    plt.figure(figsize=(10, 10))
    
    # 如果有原始图像，先显示它
    if original_image is not None:
        if isinstance(original_image, torch.Tensor):
            img = original_image.permute(1, 2, 0).cpu().numpy()
            # 归一化图像
            img = (img - img.min()) / (img.max() - img.min())
        else:
            img = original_image
            
        plt.imshow(img)
    
    # 叠加概率热力图
    plt.imshow(prob_map_smooth, cmap=cmap, alpha=alpha)
    plt.colorbar(label='概率值')
    plt.title(title)
    
    if save_path:
        # 确保保存目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()

def visualize_predictions(images, seg_preds, edge_preds, skel_preds, targets, epoch, save_dir):
    """
    可视化模型预测结果
    
    Args:
        images: 输入图像 [B,C,H,W]
        seg_preds: 分割预测 [B,1,H,W]
        edge_preds: 边缘预测 [B,1,H,W]
        skel_preds: 骨架预测 [B,1,H,W]
        targets: 元组 (seg_gt, edge_gt, skel_gt)
        epoch: 当前轮次
        save_dir: 保存目录
    """
    batch_size = images.shape[0]
    seg_gt, edge_gt, skel_gt = targets
    
    # 创建保存目录
    save_path = os.path.join(save_dir, f'epoch_{epoch}')
    os.makedirs(save_path, exist_ok=True)
    
    for i in range(min(batch_size, 4)):  # 最多显示4个样本
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        
        # 原始图像
        img = denormalize_image(images[i])
        img = img.transpose(1, 2, 0)  # [C,H,W] -> [H,W,C]
        axes[0,0].imshow(img)
        axes[0,0].set_title('输入图像')
        axes[0,0].axis('off')
        
        # 分割结果
        axes[0,1].imshow(seg_preds[i,0].cpu().numpy(), cmap='gray', vmin=0, vmax=1)
        axes[0,1].set_title('分割预测')
        axes[0,1].axis('off')
        axes[1,1].imshow(seg_gt[i,0].cpu().numpy(), cmap='gray', vmin=0, vmax=1)
        axes[1,1].set_title('分割真值')
        axes[1,1].axis('off')
        
        # 边缘结果
        axes[0,2].imshow(edge_preds[i,0].cpu().numpy(), cmap='gray', vmin=0, vmax=1)
        axes[0,2].set_title('边缘预测')
        axes[0,2].axis('off')
        axes[1,2].imshow(edge_gt[i,0].cpu().numpy(), cmap='gray', vmin=0, vmax=1)
        axes[1,2].set_title('边缘真值')
        axes[1,2].axis('off')
        
        # 骨架结果
        axes[0,3].imshow(skel_preds[i,0].cpu().numpy(), cmap='gray', vmin=0, vmax=1)
        axes[0,3].set_title('骨架预测')
        axes[0,3].axis('off')
        axes[1,3].imshow(skel_gt[i,0].cpu().numpy(), cmap='gray', vmin=0, vmax=1)
        axes[1,3].set_title('骨架真值')
        axes[1,3].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f'sample_{i}.png'), dpi=200, bbox_inches='tight')
        plt.close()

def visualize_crack_properties(predictions, original_image, save_dir):
    """
    可视化裂纹的各种属性，包括与原始图像的叠加效果
    
    Args:
        predictions: 模型预测结果字典
        original_image: 原始图像张量 [3,H,W]
        save_dir: 保存目录
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. 生长概率热力图叠加
    if 'growth_probability' in predictions:
        growth_prob = predictions['growth_probability'][0,0]
        visualize_probability_map(
            growth_prob,
            original_image,
            os.path.join(save_dir, 'growth_probability.png'),
            title='裂纹生长概率分布'
        )
    
    # 2. 矢量场叠加
    if 'velocity_field' in predictions:
        velocity_field = predictions['velocity_field'][0]
        visualize_vector_field(
            velocity_field,
            original_image,
            os.path.join(save_dir, 'velocity_field.png')
        )
    
    # 3. 拓扑点分布可视化
    if 'topology_points' in predictions:
        topo_points = predictions['topology_points'][0].cpu()
        
        # 可视化端点分布
        visualize_probability_map(
            topo_points[0],
            original_image,
            os.path.join(save_dir, 'endpoint_distribution.png'),
            title='端点分布',
            cmap='hot'
        )
        
        # 可视化交叉点分布
        visualize_probability_map(
            topo_points[1],
            original_image,
            os.path.join(save_dir, 'junction_distribution.png'),
            title='交叉点分布',
            cmap='hot'
        )
    
    # 4. 融合结果可视化
    if 'fused' in predictions:
        fused = torch.sigmoid(predictions['fused'][0,0])
        visualize_probability_map(
            fused,
            original_image,
            os.path.join(save_dir, 'fused_prediction.png'),
            title='融合预测结果'
        )

def visualize_features(model, images, layer_names, epoch, save_dir):
    """
    可视化模型中间特征图
    
    Args:
        model: 模型实例
        images: 输入图像
        layer_names: 要可视化的层名称列表
        epoch: 当前轮次
        save_dir: 保存目录
    """
    model.eval()
    activation = {}
    
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook
    
    # 注册钩子
    hooks = []
    for name in layer_names:
        layer = dict([*model.named_modules()])[name]
        hooks.append(layer.register_forward_hook(get_activation(name)))
    
    # 前向传播
    with torch.no_grad():
        _ = model(images)
    
    # 移除钩子
    for hook in hooks:
        hook.remove()
    
    # 创建保存目录
    save_path = os.path.join(save_dir, f'epoch_{epoch}', 'features')
    os.makedirs(save_path, exist_ok=True)
    
    # 可视化每一层的特征图
    for name, feat in activation.items():
        # 创建特征图网格视图
        num_channels = min(feat.shape[1], 64)  # 最多显示64个通道
        grid_tensor = make_grid(feat[0,:num_channels].unsqueeze(1), 
                            normalize=True, nrow=8)
        plt.figure(figsize=(20,20))
        plt.imshow(grid_tensor.cpu().numpy().transpose(1,2,0), cmap='viridis')
        plt.title(f'{name} - 特征图网格')
        plt.axis('off')
        plt.savefig(os.path.join(save_path, f'{name}_grid.png'), 
                dpi=200, bbox_inches='tight')
        plt.close()
# 添加演化预测和拓扑预测的可视化函数
def visualize_evolution_prediction(predictions, original_image, save_dir):
    """
    可视化演化预测结果
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. 生长概率可视化
    growth_prob = predictions['growth_probability'][0,0].cpu()
    visualize_probability_map(
        growth_prob,
        original_image,
        os.path.join(save_dir, 'growth_probability.png'),
        title='裂纹生长概率分布'
    )
    
    # 2. 速度场可视化
    velocity_field = predictions['velocity_field'][0].cpu()
    visualize_vector_field(
        velocity_field,
        original_image,
        os.path.join(save_dir, 'velocity_field.png')
    )
    
    # 3. 融合结果可视化（如果存在）
    if 'fused' in predictions:
        fused = torch.sigmoid(predictions['fused'][0,0]).cpu()
        visualize_probability_map(
            fused,
            original_image,
            os.path.join(save_dir, 'fused_prediction.png'),
            title='融合预测结果'
        )

def visualize_topology_prediction(predictions, original_image, save_dir):
    """
    可视化拓扑预测结果
    
    Args:
        predictions: 包含topology_points的预测字典
        original_image: 原始图像
        save_dir: 保存目录
    """
    os.makedirs(save_dir, exist_ok=True)
    
    if 'topology_points' in predictions:
        topo_points = predictions['topology_points'][0].cpu()
        
        # 端点预测
        visualize_probability_map(
            topo_points[0],
            original_image,
            os.path.join(save_dir, 'endpoints.png'),
            title='端点预测分布',
            cmap='cool'
        )
        
        # 交叉点预测
        if topo_points.shape[0] > 1:
            visualize_probability_map(
                topo_points[1],
                original_image,
                os.path.join(save_dir, 'junctions.png'),
                title='交叉点预测分布',
                cmap='hot'
            )