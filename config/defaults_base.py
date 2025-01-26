from yacs.config import CfgNode as CN

# 用于管理 baseline 模型的训练与测试设置。

# -----------------------------------------------------------------------------
# Convention about Training / Test specific parameters
# -----------------------------------------------------------------------------
# Whenever an argument can be either used for training or for testing, the
# corresponding name will be post-fixed by a _TRAIN for a training parameter,
# 当一个参数可以用于训练或者测试时，训练参数名称的后缀为 _TRAIN，测试参数的名称后缀为 _TEST


# -----------------------------------------------------------------------------
# Config definition 
# 配置定义
# -----------------------------------------------------------------------------

# 创建一个配置节点
_C = CN()
# -----------------------------------------------------------------------------
# MODEL 模型配置
# -----------------------------------------------------------------------------
_C.MODEL = CN()
# Using cuda or cpu for training  
# 训练时可以是 使用cuda或cpu
_C.MODEL.DEVICE = "cuda"
# ID number of GPU
# GPU ID 号
_C.MODEL.DEVICE_ID = '0'
# Name of backbone
# 主干网络名称
_C.MODEL.NAME = 'resnet50'
# Last stride of backbone
# 主干网络的最后一个步长
_C.MODEL.LAST_STRIDE = 1
# Path to pretrained model of backbone
# 主干网络预训练模型的路径
_C.MODEL.PRETRAIN_PATH = ''

# Use ImageNet pretrained model to initialize backbone or use self trained model to initialize the whole model
# Options: 'imagenet' , 'self' , 'finetune'
# 使用 ImageNet 预训练的模型初始化主干网络，或者使用自训练模型初始化整个模型
_C.MODEL.PRETRAIN_CHOICE = 'imagenet'

# If train with BNNeck, options: 'bnneck' or 'no'
# 是否使用 bnnneck 进行训练
_C.MODEL.NECK = 'bnneck'
# If train loss include center loss, options: 'yes' or 'no'. Loss with center loss has different optimizer configuration
# 训练损失函数是否包含中心损失，中心损失有不同的优化器配置

_C.MODEL.IF_WITH_CENTER = 'no'

_C.MODEL.ID_LOSS_TYPE = 'softmax'
_C.MODEL.ID_LOSS_WEIGHT = 1.0
_C.MODEL.TRIPLET_LOSS_WEIGHT = 1.0
_C.MODEL.I2T_LOSS_WEIGHT = 1.0

_C.MODEL.METRIC_LOSS_TYPE = 'triplet'
# If train with multi-gpu ddp mode, options: 'True', 'False'
# 多 GPU 分布式训练
_C.MODEL.DIST_TRAIN = False
# If train with soft triplet loss, options: 'True', 'False'
# 使用软三元组损失函数
_C.MODEL.NO_MARGIN = False
# If train with label smooth, options: 'on', 'off'
# 使用标签平滑
_C.MODEL.IF_LABELSMOOTH = 'on'
# If train with arcface loss, options: 'True', 'False'
# 使用 ArcFace 损失
_C.MODEL.COS_LAYER = False

# Transformer setting
_C.MODEL.DROP_PATH = 0.1
_C.MODEL.DROP_OUT = 0.0
_C.MODEL.ATT_DROP_RATE = 0.0
_C.MODEL.TRANSFORMER_TYPE = 'None'
_C.MODEL.STRIDE_SIZE = [16, 16]

# SIE Parameter
_C.MODEL.SIE_COE = 3.0
_C.MODEL.SIE_CAMERA = False
_C.MODEL.SIE_VIEW = False

# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
_C.INPUT = CN()
# Size of the image during training
_C.INPUT.SIZE_TRAIN = [384, 128]
# Size of the image during test
_C.INPUT.SIZE_TEST = [384, 128]
# Random probability for image horizontal flip
_C.INPUT.PROB = 0.5
# Random probability for random erasing
_C.INPUT.RE_PROB = 0.5
# Values to be used for image normalization
_C.INPUT.PIXEL_MEAN = [0.485, 0.456, 0.406]
# Values to be used for image normalization
_C.INPUT.PIXEL_STD = [0.229, 0.224, 0.225]
# Value of padding size
_C.INPUT.PADDING = 10

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASETS = CN()
# List of the dataset names for training, as present in paths_catalog.py
_C.DATASETS.NAMES = ('market1501')
# Root directory where datasets should be used (and downloaded if not found)
# 数据集的根目录，若未找到则下载
_C.DATASETS.ROOT_DIR = ('../data')


# -----------------------------------------------------------------------------
# DataLoader  数据加载器配置
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
# Number of data loading threads
_C.DATALOADER.NUM_WORKERS = 8
# Sampler for data loading
_C.DATALOADER.SAMPLER = 'softmax'
# Number of instance for one batch
_C.DATALOADER.NUM_INSTANCE = 16

# ---------------------------------------------------------------------------- #
# Solver  优化器配置
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()
# Name of optimizer
_C.SOLVER.OPTIMIZER_NAME = "Adam"
# Number of max epoches
_C.SOLVER.MAX_EPOCHS = 100
# Base learning rate
_C.SOLVER.BASE_LR = 3e-4
# Whether using larger learning rate for fc layer
_C.SOLVER.LARGE_FC_LR = False
# Factor of learning bias
_C.SOLVER.BIAS_LR_FACTOR = 1
# Factor of learning bias
_C.SOLVER.SEED = 1234
# Momentum
_C.SOLVER.MOMENTUM = 0.9
# Margin of triplet loss
_C.SOLVER.MARGIN = 0.3
# Learning rate of SGD to learn the centers of center loss
_C.SOLVER.CENTER_LR = 0.5
# Balanced weight of center loss
_C.SOLVER.CENTER_LOSS_WEIGHT = 0.0005

# Settings of weight decay
_C.SOLVER.WEIGHT_DECAY = 0.0005
_C.SOLVER.WEIGHT_DECAY_BIAS = 0.0005

# decay rate of learning rate 学习率的衰减率
_C.SOLVER.GAMMA = 0.1
# decay step of learning rate 学习率衰减步骤
_C.SOLVER.STEPS = (40, 70)
# warm up factor 预热因子
_C.SOLVER.WARMUP_FACTOR = 0.01
#  warm up epochs 预热周期数
_C.SOLVER.WARMUP_EPOCHS = 5
# 预热初始学习率
_C.SOLVER.WARMUP_LR_INIT = 0.01
# 最小学习率
_C.SOLVER.LR_MIN = 0.000016

# 预热迭代次数
_C.SOLVER.WARMUP_ITERS = 500
# method of warm up, option: 'constant','linear' 预热方法
_C.SOLVER.WARMUP_METHOD = "linear"

_C.SOLVER.COSINE_MARGIN = 0.5
_C.SOLVER.COSINE_SCALE = 30

# epoch number of saving checkpoints 保存检查点的周期
_C.SOLVER.CHECKPOINT_PERIOD = 10
# iteration of display training log 记录训练日志的迭代数
_C.SOLVER.LOG_PERIOD = 100
# epoch number of validation 验证周期数
_C.SOLVER.EVAL_PERIOD = 10
# Number of images per batch
# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 128, each GPU will
# contain 16 images per batch
# 每批次的图像数
# 全局设置，如果我们有8个GPU并且每批次图像数量是128，那么每个GPU16张图像
_C.SOLVER.IMS_PER_BATCH = 64

# ---------------------------------------------------------------------------- #
# TEST
# ---------------------------------------------------------------------------- #

_C.TEST = CN()
# Number of images per batch during test
_C.TEST.IMS_PER_BATCH = 128
# If test with re-ranking, options: 'True','False'
# 是否使用重新排序
_C.TEST.RE_RANKING = False
# Path to trained model
# 训练模型的路径
_C.TEST.WEIGHT = ""
# Which feature of BNNeck to be used for test, before or after BNNneck, options: 'before' or 'after'
# 使用哪一个BNNECK特征
_C.TEST.NECK_FEAT = 'after'
# Whether feature is nomalized before test, if yes, it is equivalent to cosine distance
# 测试前是否归一化特征
_C.TEST.FEAT_NORM = 'yes'

# Name for saving the distmat after testing.
# 测试后保存的距离矩阵的名字
_C.TEST.DIST_MAT = "dist_mat.npy"
# Whether calculate the eval score option: 'True', 'False'
# 是否计算评估分数
_C.TEST.EVAL = False
# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
# Path to checkpoint and saved log of trained model
# 检查点与训练日志的保存路径
_C.OUTPUT_DIR = ""
