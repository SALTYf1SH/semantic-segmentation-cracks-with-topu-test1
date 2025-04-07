import os

class Config:
    # 数据集配置
    DATA_DIR = "G:\\语义分割裂隙\\data\\data\\"  # 修改为包含所有图像和标签的父目录
    
    # 决定是否使用分割好的训练/验证集目录
    USE_SPLIT_FOLDERS = True  # 设置为False将使用同一目录
    USE_AMP = True  # 启用自动混合精度训练
    
    # 基础图像和标签目录
    IMAGE_DIR = os.path.join(DATA_DIR, "raw")  # 所有图像的目录
    LABEL_DIR = os.path.join(DATA_DIR, "label")  # 所有标签的目录
    
    VIS_DIR = "visualizations"
    VIS_SAVE_INTERVAL = 1  # 每隔多少个epoch保存一次可视化结果
    MAX_SAMPLES_PER_BATCH = 4  # 每个batch最多可视化多少个样本
    FEATURE_VIS_CHANNELS = 64  # 特征图可视化时最多显示多少个通道
    

    # 训练集和验证集配置
    def __init__(self):
        if self.USE_SPLIT_FOLDERS:
            # 使用分好的训练集和验证集
            self.TRAIN_IMAGE_DIR = os.path.join(self.DATA_DIR, "train", "raw")
            self.TRAIN_LABEL_DIR = os.path.join(self.DATA_DIR, "train", "label")
            self.VAL_IMAGE_DIR = os.path.join(self.DATA_DIR, "val", "raw")
            self.VAL_LABEL_DIR = os.path.join(self.DATA_DIR, "val", "label")
        else:
            # 使用同一目录（需要在dataloader中手动分割）
            self.TRAIN_IMAGE_DIR = self.IMAGE_DIR
            self.TRAIN_LABEL_DIR = self.LABEL_DIR
            self.VAL_IMAGE_DIR = self.IMAGE_DIR
            self.VAL_LABEL_DIR = self.LABEL_DIR
        
        # 测试集目录（修改为绝对路径）
        self.TEST_IMAGE_DIR = os.path.join(self.DATA_DIR, "test", "images")
        self.TEST_LABEL_DIR = os.path.join(self.DATA_DIR, "test", "labels")
    
    # 输出与可视化目录
    CHECKPOINT_DIR = "checkpoints"
    LOG_DIR = "logs"
    VIS_DIR = "visualizations"
    
    # 训练参数
    BATCH_SIZE = 4
    NUM_EPOCHS = 100
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5
    DEVICE = "cuda"
    SAVE_FREQ = 5  # 每多少个epoch保存一次模型
    PATIENCE = 10  # 早停等待周期
    VIS_INTERVAL = 2  # 可视化间隔(epoch)

    # 模型参数
    INPUT_CHANNELS = 3
    OUTPUT_CHANNELS = 1

    # 损失函数权重
    ALPHA = 0.6  # 分割损失权重
    BETA = 0.2   # 边缘损失权重
    GAMMA = 0.2  # 拓扑损失权重
    
    POSTPROCESSING_THRESHOLD = 0.5  # 二值化阈值
    POSTPROCESSING_MIN_AREA = 10    # 最小连通区域面积
    SAVE_POSTPROCESSING = True      # 是否保存后处理结果
    def setup_dirs(self):
        """创建必要的目录结构"""
        vis_dirs = [
            os.path.join(self.VIS_DIR, 'validation'),
            os.path.join(self.VIS_DIR, 'features')
        ]
        for d in vis_dirs:
            os.makedirs(d, exist_ok=True)
        
        # 检查数据目录是否存在
        required_data_dirs = [
            self.TRAIN_IMAGE_DIR, self.TRAIN_LABEL_DIR,
            self.VAL_IMAGE_DIR, self.VAL_LABEL_DIR,
            self.TEST_IMAGE_DIR, self.TEST_LABEL_DIR
        ]
        
        missing_dirs = [d for d in required_data_dirs if not os.path.exists(d)]
        if missing_dirs:
            print("警告: 以下数据目录不存在:")
            for d in missing_dirs:
                print(f"  - {d}")
            print("请确保这些目录存在或修改配置")