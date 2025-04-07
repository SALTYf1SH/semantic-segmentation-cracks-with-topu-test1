import torch
import torch.nn as nn

class DifferentiableTopology(nn.Module):
    """可微分拓扑特征提取层"""
    def __init__(self, sigma=1.0):
        super().__init__()
        self.sigma = sigma
        
        # 定义可学习参数
        self.kernel = nn.Parameter(torch.tensor([
            [0.5, 1.0, 0.5],
            [1.0, -6., 1.0],
            [0.5, 1.0, 0.5]
        ], dtype=torch.float32), requires_grad=False)  # 固定骨架检测卷积核

    def forward(self, x):
        """
        输入: [B,1,H,W] 分割概率图
        输出: [B,1,H,W] 拓扑特征图
        """
        batch_size = x.shape[0]
        
        # 步骤1：可微分二值化
        binary = torch.sigmoid(10*(x-0.5))  # 近似阶跃函数
        
        # 步骤2：可微分骨架化
        skeleton = self.differentiable_skeleton(binary)
        
        # 步骤3：端点检测
        endpoints = self.find_endpoints(skeleton)
        
        return endpoints * skeleton  # 组合特征

    def differentiable_skeleton(self, x):
        """可微分骨架化近似"""
        # 使用固定卷积核检测中心线
        conv = torch.nn.functional.conv2d(
            x, 
            self.kernel[None, None, ...].to(x.device), 
            padding=1
        )
        
        # 近似非极大值抑制
        skeleton = torch.sigmoid(self.sigma * (conv + 2))  # 正值响应区域
        return skeleton

    def find_endpoints(self, x):
        """可微分端点检测"""
        # 使用3x3卷积计算邻域和
        neighbor_kernel = torch.ones(1,1,3,3).to(x.device) / 8.0
        neighbor_sum = torch.nn.functional.conv2d(
            x, neighbor_kernel, padding=1
        )
        
        # 端点条件：中心为1且周围恰好1个邻居
        endpoints = x * torch.sigmoid(10*(1.5 - neighbor_sum))  # 0.5~2.5范围平滑过渡
        return endpoints