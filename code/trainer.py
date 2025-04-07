import torch
import numpy as np
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from visualization import (
    visualize_feature_heatmap, 
    visualize_evolution_prediction, 
    visualize_topology_prediction,
    visualize_predictions, 
    visualize_features
)
from Postprocessing import FracturePostProcessor
from vector_analysis import CrackVectorAnalyzer

plt.rcParams['font.sans-serif'] = ['SimSun']  # 使用宋体
plt.rcParams['axes.unicode_minus'] = False     # 解决负号显示问题

class Trainer:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, config, lr_scheduler=None):
        """
        初始化训练器
        
        Args:
            model: 模型实例
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            criterion: 损失函数
            optimizer: 优化器
            config: 配置对象
            lr_scheduler: 学习率调度器（可选）
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.config = config
        self.lr_scheduler = lr_scheduler  # 添加学习率调度器
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.writer = SummaryWriter(log_dir=config.LOG_DIR)
        self.current_epoch = 0 
        
        # 自动混合精度训练
        self.scaler = torch.cuda.amp.GradScaler()
        
        # 初始化后处理器和矢量分析器
        self.postprocessor = FracturePostProcessor(threshold=0.5, min_area=10)
        self.vector_analyzer = CrackVectorAnalyzer()
        
        self.model.to(self.device)

    def train(self):
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.config.NUM_EPOCHS):
            self.current_epoch = epoch  # 更新当前epoch
            print(f"\nEpoch {epoch+1}/{self.config.NUM_EPOCHS}")
            
            # 训练阶段
            train_loss = self.train_epoch()
            
            # 验证阶段
            val_loss = self.validate()
            
            # 记录到TensorBoard
            self.writer.add_scalars('Loss', {
                'train': train_loss,
                'val': val_loss
            }, epoch)
            
            # 学习率调整
            if self.lr_scheduler is not None:
                self.lr_scheduler.step(val_loss)
            
            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.save_checkpoint(
                    os.path.join(self.config.CHECKPOINT_DIR, 'best_model.pth'),
                    epoch, val_loss
                )
                print(f"New best model saved with val_loss: {val_loss:.4f}")
            else:
                patience_counter += 1
                
            # 定期保存
            if (epoch + 1) % self.config.SAVE_FREQ == 0:
                self.save_checkpoint(
                    os.path.join(self.config.CHECKPOINT_DIR, f'checkpoint_epoch_{epoch+1}.pth'),
                    epoch, val_loss
                )
            
            # 早停
            if patience_counter >= self.config.PATIENCE:
                print(f"\nEarly stopping triggered after {epoch+1} epochs!")
                break
    def save_checkpoint(self, path, epoch, val_loss):
        """保存检查点"""
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'lr_scheduler_state_dict': self.lr_scheduler.state_dict() if self.lr_scheduler else None,
            'val_loss': val_loss,
        }, path)

    def train_epoch(self):
        self.model.train()
        total_loss = 0.0
        progress_bar = tqdm(self.train_loader, desc="Training")
        
        for batch in progress_bar:
            # 解包batch元组
            images, seg_labels, edge_labels, skel_labels = batch
            
            # 将数据移动到设备
            images = images.to(self.device)
            seg_labels = seg_labels.to(self.device)
            edge_labels = edge_labels.to(self.device)
            skel_labels = skel_labels.to(self.device)
            
            with torch.cuda.amp.autocast(enabled=self.config.USE_AMP):
                # 前向传播
                predictions = self.model(images)
                
                if isinstance(predictions, dict):
                    seg_pred = predictions['segmentation']
                    edge_pred = predictions['edges']
                    skel_pred = predictions['skeleton']
                else:
                    seg_pred, edge_pred, skel_pred = predictions
                
                # 构建目标字典以匹配损失函数期望的格式
                targets = {
                    'current_mask': seg_labels,
                    'current_edge': edge_labels,
                    'current_skeleton': skel_labels,
                }
                
                # 计算损失
                loss = self.criterion(predictions, targets)
            
            # 反向传播
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        return total_loss / len(self.train_loader)

    def validate(self):
        self.model.eval()
        total_loss = 0.0
        progress_bar = tqdm(self.val_loader, desc="Validation")
        
        with torch.no_grad():
            prev_images = None
            prev_topo_results = None  # 存储前一帧的拓扑结果
            
            for idx, (images, seg_labels, edge_labels, skel_labels) in enumerate(progress_bar):
                images = images.to(self.device)
                seg_labels = seg_labels.to(self.device)
                edge_labels = edge_labels.to(self.device)
                skel_labels = skel_labels.to(self.device)
                
                # 模型预测
                predictions = self.model(images, prev_frame=prev_images)
                
                # 构建目标字典
                targets = {
                    'current_mask': seg_labels,
                    'current_edge': edge_labels,
                    'current_skeleton': skel_labels,
                }
                
                # 计算损失
                loss = self.criterion(predictions, targets)
                
                total_loss += loss.item()
                progress_bar.set_postfix({'val_loss': f"{loss.item():.4f}"})
                
                # 对预测结果进行后处理
                # 对预测结果进行后处理
                seg_prob = torch.sigmoid(predictions['segmentation'])
                
                # 对批次中的第一个样本进行后处理和可视化
                post_results = []
                for i in range(min(images.size(0), 4)):  # 最多处理4个样本
                    # 后处理
                    post_result = self.postprocessor.process(
                        seg_prob[i].detach().cpu(),
                        torch.sigmoid(predictions['edges'])[i].detach().cpu() if 'edges' in predictions else None
                    )
                    post_results.append(post_result)
                
                # 可视化（每个epoch保存几个样本）
                if idx % self.config.VIS_SAVE_INTERVAL == 0:
                    batch_vis_dir = os.path.join(
                        self.config.VIS_DIR, 
                        'validation', 
                        f'epoch_{self.current_epoch}',
                        f'batch_{idx}'
                    )
                    os.makedirs(batch_vis_dir, exist_ok=True)
                    
                    # 1. 基础分割结果可视化
                    visualize_predictions(
                        images,
                        seg_prob,
                        torch.sigmoid(predictions['edges']) if 'edges' in predictions else None,
                        predictions['skeleton'] if 'skeleton' in predictions else None,
                        (seg_labels, edge_labels, skel_labels),
                        self.current_epoch,
                        batch_vis_dir
                    )
                    
                    # 2. 后处理结果可视化
                    for i, post_result in enumerate(post_results):
                        # 保存后处理可视化
                        post_vis_path = os.path.join(batch_vis_dir, f'postprocessing_sample_{i}.png')
                        self.postprocessor.visualize(post_result, save_path=post_vis_path)
                    
                    # 3. 演化预测和拓扑预测可视化
                    if predictions.get('velocity_field') is not None:
                        # 演化预测
                        visualize_evolution_prediction(
                            predictions,
                            images[0],
                            os.path.join(batch_vis_dir, 'evolution')
                        )
                        
                        # 拓扑预测
                        visualize_topology_prediction(
                            predictions,
                            images[0],
                            os.path.join(batch_vis_dir, 'topology')
                        )
                        
                        # 矢量分析 - 修复这里的条件判断
                        if len(post_results) > 0 and prev_topo_results is not None:
                            # 分析生长方向
                            growth_analysis = self.vector_analyzer.analyze_growth_direction(
                                predictions['velocity_field'][0].cpu().numpy(),
                                predictions['growth_probability'][0,0].cpu().numpy()
                            )
                            
                            # 分析拓扑变化
                            if 'topology_points' in predictions:
                                # 创建拓扑分析保存目录
                                topo_analysis_dir = os.path.join(batch_vis_dir, 'topo_analysis')
                                os.makedirs(topo_analysis_dir, exist_ok=True)
                                
                                # 保存生长方向分析结果
                                self.vector_analyzer.visualize_analysis(
                                    growth_analysis, 
                                    topo_analysis_dir
                                )
                    
                    # 4. 特征可视化
                    with torch.no_grad():
                        features = self.model.extract_features(images)
                        if isinstance(features, torch.Tensor):
                            # 只处理第一个样本的特征
                            sample_features = features[0]
                            feature_vis_path = os.path.join(batch_vis_dir, 'feature_heatmap.png')
                            visualize_feature_heatmap(sample_features, feature_vis_path)
                
                # 保存当前预测供下一帧使用
                prev_images = images.detach().clone()
                if 'topology_points' in predictions:
                    prev_topo_results = predictions['topology_points'][0].detach().cpu().numpy()
        
        return total_loss / len(self.val_loader)

    def save_model(self, filename):
        save_path = os.path.join(self.config.MODEL_SAVE_DIR, filename)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, save_path)