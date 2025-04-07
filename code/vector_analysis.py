import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt

class CrackVectorAnalyzer:
    def __init__(self):
        pass
    
    def analyze_growth_direction(self, velocity_field, growth_prob):
        """分析裂纹生长方向"""
        vx, vy = velocity_field[0], velocity_field[1]
        magnitude = np.sqrt(vx**2 + vy**2)
        angle = np.arctan2(vy, vx)
        
        # 主要生长方向
        mask = growth_prob > 0.5
        if mask.sum() > 0:
            mean_angle = np.mean(angle[mask])
            mean_magnitude = np.mean(magnitude[mask])
        else:
            mean_angle = 0
            mean_magnitude = 0
            
        return {
            'mean_angle': mean_angle,
            'mean_magnitude': mean_magnitude,
            'angle_map': angle,
            'magnitude_map': magnitude
        }
    
    def analyze_topology_evolution(self, prev_topo, curr_topo):
        """分析拓扑结构的演化"""
        # 端点变化
        new_endpoints = curr_topo[0] > prev_topo[0]
        # 交叉点变化
        new_junctions = curr_topo[1] > prev_topo[1]
        
        return {
            'new_endpoints': new_endpoints,
            'new_junctions': new_junctions,
            'n_new_endpoints': np.sum(new_endpoints),
            'n_new_junctions': np.sum(new_junctions)
        }
    
    def visualize_analysis(self, analysis_results, save_dir):
        """可视化分析结果"""
        # 1. 生长方向分布
        plt.figure(figsize=(10, 10))
        plt.imshow(analysis_results['angle_map'], cmap='hsv')
        plt.colorbar(label='Growth Direction (rad)')
        plt.title(f"Mean Direction: {analysis_results['mean_angle']:.2f} rad")
        plt.savefig(f"{save_dir}/growth_direction.png")
        plt.close()
        
        # 2. 速度分布
        plt.figure(figsize=(10, 10))
        plt.imshow(analysis_results['magnitude_map'], cmap='viridis')
        plt.colorbar(label='Growth Speed')
        plt.title(f"Mean Speed: {analysis_results['mean_magnitude']:.2f}")
        plt.savefig(f"{save_dir}/growth_speed.png")
        plt.close()