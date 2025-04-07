import cv2
import numpy as np
import torch
import networkx as nx
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize, dilation, disk
from skimage import img_as_ubyte
from skimage.morphology import closing
from skan import Skeleton, summarize  # 新增：引入skan库

class FracturePostProcessor:
    def __init__(self, threshold=0.5, min_area=10):
        """
        后处理模块，处理模型输出并提取拓扑网络
        
        Args:
            threshold: 二值化阈值
            min_area: 最小连通区域面积，小于此面积的区域将被移除
        """
        self.threshold = threshold
        self.min_area = min_area
    
    def process(self, seg_pred, edge_pred=None, original_image=None):
        """
        处理模型输出，生成二值掩码、骨架和拓扑图结构
        
        Args:
            seg_pred: 分割掩码
            edge_pred: 边缘预测 (可选)
            original_image: 原始图像 (可选)
            
        Returns:
            dict: 包含处理结果的字典
        """
        # 转换为 numpy 数组
        if isinstance(seg_pred, torch.Tensor):
            seg_pred = seg_pred.detach().cpu().numpy()
        
        # 确保形状正确
        if len(seg_pred.shape) == 4:  # [B,C,H,W]
            seg_pred = seg_pred[0, 0]  # 取第一个样本的第一个通道
        elif len(seg_pred.shape) == 3:  # [C,H,W]
            seg_pred = seg_pred[0]  # 取第一个通道
        
        # 二值化
        binary_mask = (seg_pred > self.threshold).astype(np.uint8) * 255
        
        # 形态学操作增强裂隙连通性
        binary_mask = dilation(binary_mask, disk(1))
        binary_mask = closing(binary_mask, disk(2))
        
        # 移除小连通区域
        if self.min_area > 0:
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
            for i in range(1, num_labels):  # 跳过背景
                if stats[i, cv2.CC_STAT_AREA] < self.min_area:
                    binary_mask[labels == i] = 0
        
        # 提取骨架
        binary_bool = binary_mask > 0
        skeleton = skeletonize(binary_bool)
        skeleton_img = img_as_ubyte(skeleton)
        
        # 使用skan提取拓扑结构
        graph = self.extract_topology(skeleton)
        
        # 返回结果
        return {
            'binary_mask': binary_mask,
            'skeleton': skeleton_img,
            'graph': graph
        }
    
    def extract_topology(self, skeleton):
        """
        从骨架图中提取拓扑结构（节点和边）
        
        Args:
            skeleton: 骨架图（二值化）
        
        Returns:
            graph: 包含节点和边的图结构
        """
        # 初始化网络图
        G = nx.Graph()
        
        # 确保骨架中有非零像素
        if np.sum(skeleton) == 0:
            print("警告: 骨架为空。未检测到裂隙或裂隙过细。")
    
        try:
            # 使用skan分析骨架
            skel_obj = Skeleton(skeleton)
            branch_data = summarize(skel_obj, separator='_')
            
            # 提取节点坐标
            node_coords = skel_obj.coordinates
            num_coords = len(node_coords)
            
            # 获取有效的节点ID范围
            valid_src_ids = branch_data['node_id_src'][branch_data['node_id_src'] < num_coords]
            valid_dst_ids = branch_data['node_id_dst'][branch_data['node_id_dst'] < num_coords]
            unique_node_ids = np.unique(np.concatenate([valid_src_ids, valid_dst_ids]))
            
            # 添加节点到图中
            for node_id in unique_node_ids:
                node_idx = int(node_id)
                if 0 <= node_idx < num_coords:  # 确保索引在有效范围内
                    r, c = node_coords[node_idx]
                    r, c = int(r), int(c)
                    
                    # 确定节点类型 (端点或交叉点)
                    node_type = 'junction'  # 默认为交叉点
                    if (np.sum(branch_data['node_id_src'] == node_id) + 
                        np.sum(branch_data['node_id_dst'] == node_id)) == 1:
                        node_type = 'endpoint'  # 只有一条边连接的是端点
                    
                    # 添加节点
                    G.add_node(node_idx, pos=(r, c), type=node_type)
            
            # 添加边到图中，使用过滤后的数据
            for _, row in branch_data.iterrows():
                src_node_idx = int(row['node_id_src'])
                dst_node_idx = int(row['node_id_dst'])
                
                # 检查节点索引是否在有效范围内
                if (0 <= src_node_idx < num_coords and 
                    0 <= dst_node_idx < num_coords and 
                    src_node_idx in G and 
                    dst_node_idx in G):
                    
                    length = row['branch_distance']
                    branch_id = row['branch_id'] if 'branch_id' in branch_data.columns else row.name
                    
                    try:
                        # 获取边的路径坐标
                        path_indices = skel_obj.path_coordinates(branch_id)
                        # 过滤无效的路径索引
                        valid_indices = path_indices[path_indices < num_coords]
                        if len(valid_indices) > 0:
                            path_coords = node_coords[valid_indices].tolist()
                            # 添加边
                            G.add_edge(src_node_idx, dst_node_idx, 
                                    weight=length, path=path_coords)
                    except Exception as e:
                        print(f"警告: 处理边 {src_node_idx}-{dst_node_idx} 的路径时出错: {e}")
                        continue
        
        except Exception as e:
            print(f"警告: 提取拓扑结构时出错: {e}")
            # 如果出错，返回空图
            G = nx.Graph()
        
        return G
    
    def visualize(self, result, save_path=None):
        """可视化后处理结果"""
        binary_mask = result['binary_mask']
        skeleton = result['skeleton']
        graph = result['graph']
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # 显示二值掩码
        axes[0].imshow(binary_mask, cmap='gray')
        axes[0].set_title('二值掩码')
        axes[0].axis('off')
        
        # 显示骨架
        axes[1].imshow(skeleton, cmap='gray')
        axes[1].set_title('骨架')
        axes[1].axis('off')
        
        # 显示拓扑结构
        axes[2].imshow(skeleton, cmap='gray')
        axes[2].set_title('拓扑结构')
        axes[2].axis('off')
        
        # 绘制节点和边
        pos = nx.get_node_attributes(graph, 'pos')
        # 调整坐标格式，从(r,c)转为(c,r)以匹配matplotlib的坐标系
        pos_plot = {n: (p[1], p[0]) for n, p in pos.items()}
        
        # 绘制不同类型的节点
        endpoints = [n for n, d in graph.nodes(data=True) if d.get('type') == 'endpoint']
        junctions = [n for n, d in graph.nodes(data=True) if d.get('type') == 'junction']
        
        nx.draw_networkx_nodes(graph, pos=pos_plot, nodelist=endpoints, node_size=30, 
                               node_color='red', ax=axes[2])
        nx.draw_networkx_nodes(graph, pos=pos_plot, nodelist=junctions, node_size=30, 
                               node_color='green', ax=axes[2])
        nx.draw_networkx_edges(graph, pos=pos_plot, edge_color='blue', width=1.5, ax=axes[2])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"可视化结果已保存至: {save_path}")
        
        plt.close()
        
        return fig