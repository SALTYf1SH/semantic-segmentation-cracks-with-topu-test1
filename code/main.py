import os
# 在导入torch和numpy之前设置环境变量
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # 允许OpenMP多重初始化
os.environ['OMP_NUM_THREADS'] = '1'  # 限制OpenMP线程数为1
os.environ['MKL_NUM_THREADS'] = '1'  # 限制MKL线程数
os.environ['NUMEXPR_NUM_THREADS'] = '1'  # 限制numexpr线程数
import torch
from torch.utils.data import DataLoader
from model import TopoUNet
from loss import TopoLoss
from dataloader import get_dataloader
from trainer import Trainer
from config import Config
import argparse
import visualization
import os
os.environ['OMP_NUM_THREADS'] = '1'
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimSun']  # 使用宋体
plt.rcParams['axes.unicode_minus'] = False     # 解决负号显示问题


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='裂纹预测训练与评估')
    parser.add_argument('--mode', type=str, default='train', 
                      choices=['train', 'test', 'predict'],
                      help='运行模式: train(训练)/test(评估)/predict(预测)')
    parser.add_argument('--ckpt', type=str, default=None,
                      help='测试/预测时使用的模型检查点路径')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 初始化配置
    config = Config()
    config.setup_dirs()
    
    # 数据加载
    train_loader, val_loader, test_loader = get_dataloader(
        train_img_dir=config.TRAIN_IMAGE_DIR,
        train_label_dir=config.TRAIN_LABEL_DIR,
        val_img_dir=config.VAL_IMAGE_DIR,
        val_label_dir=config.VAL_LABEL_DIR,
        test_img_dir=config.TEST_IMAGE_DIR,
        test_label_dir=config.TEST_LABEL_DIR,
        batch_size=config.BATCH_SIZE
    )
    
    # 模型初始化
    model = TopoUNet()
    
    if args.mode in ['test', 'predict']:
        # 加载预训练模型
        if not args.ckpt:
            raise ValueError("测试/预测模式必须指定--ckpt参数")
        checkpoint = torch.load(args.ckpt)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"成功加载模型: {args.ckpt}")
    
    # 设备设置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    if args.mode == 'train':
        # 训练流程
        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )
        criterion = TopoLoss(
            alpha=config.ALPHA,
            beta=config.BETA,
            gamma=config.GAMMA
        )
        
        # 创建学习率调度器
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 'min', patience=3, factor=0.5, verbose=True
        )
        
        # 修改Trainer初始化，将lr_scheduler作为单独参数传入
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,  # 改用lr_scheduler作为参数名
            config=config,
        )
        trainer.train()
        
    elif args.mode == 'test':
        # 评估流程
        evaluator = Trainer(
            model=model,
            val_loader=test_loader,
            config=config,
            device=device
        )
        metrics = evaluator.evaluate(verbose=True)
        print("\n评估结果:")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")
            
    elif args.mode == 'predict':
        # 预测与可视化
        from visualization import plot_predictions
        sample_batch = next(iter(test_loader))
        with torch.no_grad():
            model.eval()
            images = sample_batch[0].to(device)
            seg_pred, edge_pred, skel_pred = model(images)
            
        plot_predictions(
            images.cpu(), 
            seg_pred.sigmoid().cpu(), 
            edge_pred.sigmoid().cpu(),
            skel_pred.cpu(),
            save_path=os.path.join(config.VIS_DIR, 'prediction_samples.png')
        )

if __name__ == '__main__':
    main()