import torch
import torch.nn as nn
import torch.nn.functional as F
from DifferentiableTopology import DifferentiableTopology

class TopoLoss(nn.Module):
    def __init__(self, alpha=0.4, beta=0.2, gamma=0.2, delta=0.2):
        super().__init__()
        self.alpha = alpha  # 分割损失权重
        self.beta = beta    # 边缘损失权重
        self.gamma = gamma  # 骨架损失权重
        self.delta = delta  # 演化损失权重
        
        self.dice_loss = DiceLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.mse_loss = nn.MSELoss()
        self.topo_layer = DifferentiableTopology()
    
    def edge_loss(self, pred, target):
        """
        计算边缘损失
        Args:
            pred: 预测的边缘图 [B,1,H,W]
            target: 目标边缘图 [B,1,H,W]
        """
        return self.bce_loss(pred, target)
    
    def skeleton_loss(self, pred, target):
        """
        计算骨架损失
        Args:
            pred: 预测的骨架图 [B,1,H,W]
            target: 目标骨架图 [B,1,H,W]
        """
        return self.bce_loss(pred, target)

    def forward(self, predictions, targets):
        # 基础分割损失
        seg_loss = self.dice_loss(torch.sigmoid(predictions['segmentation']), 
                                targets['current_mask'])
        edge_loss = self.edge_loss(predictions['edges'], 
                                 targets['current_edge'])
        skeleton_loss = self.skeleton_loss(predictions['skeleton'],
                                         targets['current_skeleton'])
        
        # 如果没有演化预测,只返回基础损失
        if predictions.get('evolution') is None:
            return (self.alpha * seg_loss + 
                   self.beta * edge_loss + 
                   self.gamma * skeleton_loss)
        
        # 演化预测损失
        growth_loss = self.bce_loss(predictions['growth_probability'],
                                  targets['growth_region'])
        
        velocity_loss = self.mse_loss(predictions['velocity_field'],
                                    targets['displacement'])
        
        topo_point_loss = self.bce_loss(predictions['topology_points'],
                                      targets['topology_changes'])
        
        # 总损失
        total_loss = (
            self.alpha * seg_loss +
            self.beta * edge_loss + 
            self.gamma * skeleton_loss +
            self.delta * (growth_loss + velocity_loss + topo_point_loss)
        )
        
        return total_loss

class DiceLoss(nn.Module):
    def forward(self, pred, target, smooth=1.0):
        """
        计算 Dice 损失
        注意：输入 pred 应该已经经过 sigmoid
        Args:
            pred: 预测概率图 [B,1,H,W]
            target: 目标掩码 [B,1,H,W]
            smooth: 平滑项
        """
        # 确保输入已经是概率值
        assert torch.all((pred >= 0) & (pred <= 1)), "Pred 必须是概率值 (应用 sigmoid)"
        
        # 计算 Dice 系数
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum()
        
        # 返回 Dice 损失
        return 1 - (2. * intersection + smooth) / (union + smooth)