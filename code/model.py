import torch
import torch.nn as nn
from torchvision.models import resnet34, ResNet34_Weights

class TopoUNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=1):
        super().__init__()
        
        # 使用新的权重参数初始化方式
        backbone = resnet34(weights=ResNet34_Weights.DEFAULT)
        
        # 编码器
        self.encoder1 = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu
        )  # [B,64,H/2,W/2]
        self.pool = backbone.maxpool  # [B,64,H/4,W/4]
        self.encoder2 = backbone.layer1  # [B,64,H/4,W/4]
        self.encoder3 = backbone.layer2  # [B,128,H/8,W/8]
        self.encoder4 = backbone.layer3  # [B,256,H/16,W/16]
        self.encoder5 = backbone.layer4  # [B,512,H/32,W/32]

        # 上采样层
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        # 解码器
        self.decoder5 = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )  # [B,256,H/32,W/32]
        
        self.decoder4 = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )  # [B,256,H/16,W/16]
        
        self.decoder3 = nn.Sequential(
            nn.Conv2d(384, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )  # [B,128,H/8,W/8]
        
        self.decoder2 = nn.Sequential(
            nn.Conv2d(192, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )  # [B,64,H/4,W/4]
        
        self.decoder1 = nn.Sequential(
            nn.Conv2d(128, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )  # [B,32,H/2,W/2]
        
        # 输出头
        self.seg_head = nn.Sequential(
            nn.Conv2d(32, num_classes, 1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  # 只上采样2倍
        )
        
        self.edge_head = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  # 从 H/2 变为 H
        )
        
        self.skel_head = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True),  # 从 H/4 变为 H
            nn.Sigmoid()
        )
        self.temporal_encoder = nn.Sequential(
            nn.Conv2d(512*2, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.evolution_predictor = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, 1),  # [growth_prob, velocity_x, velocity_y]
            nn.Upsample(scale_factor=32, mode='bilinear', align_corners=True)  # 恢复到原始尺寸
        )

        # 修改拓扑点预测器，添加上采样
        self.topology_predictor = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 2, 1),  # [endpoint_prob, junction_prob]
            nn.Upsample(scale_factor=32, mode='bilinear', align_corners=True)  # 恢复到原始尺寸
        )

        # 添加融合层
        self.fusion_layer = nn.Sequential(
            nn.Conv2d(8, 32, 3, padding=1),  # 修改输入通道为8
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_classes, 1)
        )
    def extract_vector_features(self, x):
        """
        提取与矢量信息相关的深层特征
        
        Args:
            x: 输入图像 [B,C,H,W]
        Returns:
            dict: 包含各层特征的字典
        """
        features = {}
        
        # 编码器特征
        e1 = self.encoder1(x)
        e1p = self.pool(e1)
        e2 = self.encoder2(e1p)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        e5 = self.encoder5(e4)
        
        # 存储各层特征
        features['encoder5'] = e5  # 最深层特征
        features['temporal'] = self.temporal_encoder(torch.cat([e5, e5], dim=1))  # 时序特征
        
        # 计算特征响应图
        with torch.no_grad():
            feature_response = torch.mean(torch.abs(e5), dim=1, keepdim=True)
            features['response_map'] = feature_response
        
        return features
    def extract_features(self, x):
            """提取深层特征"""
            e1 = self.encoder1(x)
            e1p = self.pool(e1)
            e2 = self.encoder2(e1p)
            e3 = self.encoder3(e2)
            e4 = self.encoder4(e3)
            e5 = self.encoder5(e4)
            return e5

    def forward(self, x, prev_frame=None):
        """
        前向传播
        Args:
            x: 当前帧 [B,C,H,W]
            prev_frame: 前一帧 (可选) [B,C,H,W]
        """
        # 提取当前帧特征
        e1 = self.encoder1(x)
        e1p = self.pool(e1)
        e2 = self.encoder2(e1p)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        e5 = self.encoder5(e4)
        current_features = e5  # [B,512,H/32,W/32]

        # 基础分割结果
        d5 = self.decoder5(e5)
        d5 = self.upsample(d5)
        
        d4 = self.decoder4(torch.cat([d5, e4], dim=1))
        d4 = self.upsample(d4)
        
        d3 = self.decoder3(torch.cat([d4, e3], dim=1))
        d3 = self.upsample(d3)
        
        d2 = self.decoder2(torch.cat([d3, e2], dim=1))
        d2 = self.upsample(d2)
        
        d1 = self.decoder1(torch.cat([d2, e1], dim=1))

        # 基础输出
        seg_pred = self.seg_head(d1)
        edge_pred = self.edge_head(d2)
        skel_pred = self.skel_head(d3)

        if prev_frame is None:
            return {
                'segmentation': seg_pred,
                'edges': edge_pred,
                'skeleton': skel_pred,
                'evolution': None,
                'topology': None,
                'fused': seg_pred  # 当没有前一帧时，直接使用分割结果
            }

        # 确保前一帧的batch size与当前帧相同
        if prev_frame.size(0) != x.size(0):
            prev_frame = prev_frame[:x.size(0)]

        # 提取前一帧特征
        with torch.no_grad():
            prev_e1 = self.encoder1(prev_frame)
            prev_e1p = self.pool(prev_e1)
            prev_e2 = self.encoder2(prev_e1p)
            prev_e3 = self.encoder3(prev_e2)
            prev_e4 = self.encoder4(prev_e3)
            prev_e5 = self.encoder5(prev_e4)
            prev_features = prev_e5  # [B,512,H/32,W/32]

        # 连接时序特征
        try:
            temporal_feat = torch.cat([current_features, prev_features], dim=1)  # [B,1024,H/32,W/32]
        except RuntimeError as e:
            print(f"当前特征尺寸: {current_features.shape}")
            print(f"前一帧特征尺寸: {prev_features.shape}")
            raise e

        # 通过时序编码器
        temporal_feat = self.temporal_encoder(temporal_feat)  # [B,256,H/32,W/32]

        # 预测演化信息
        evolution_pred = self.evolution_predictor(temporal_feat)
        growth_prob = evolution_pred[:,0:1]
        velocity_field = evolution_pred[:,1:]

        # 预测拓扑点并上采样到原始尺寸
        topo_points = self.topology_predictor(temporal_feat)

        # 融合所有输出
        fused_input = torch.cat([
            seg_pred,                    # [B,1,H,W]
            edge_pred,                   # [B,1,H,W]
            skel_pred,                   # [B,1,H,W]
            growth_prob,                 # [B,1,H,W]
            velocity_field,              # [B,2,H,W]
            topo_points                  # [B,2,H,W]
        ], dim=1)
        print(f"Fused input shape: {fused_input.shape}")
        fused_output = self.fusion_layer(fused_input)

        return {
            'segmentation': seg_pred,
            'edges': edge_pred,
            'skeleton': skel_pred,
            'growth_probability': growth_prob,
            'velocity_field': velocity_field,
            'topology_points': topo_points,
            'fused': fused_output
        }