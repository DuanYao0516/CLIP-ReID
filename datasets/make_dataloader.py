import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader

from .bases import ImageDataset
from timm.data.random_erasing import RandomErasing
from .sampler import RandomIdentitySampler

# 下面是引入的一些数据集类的构造函数
from .dukemtmcreid import DukeMTMCreID
from .market1501 import Market1501
from .msmt17 import MSMT17
from .occ_duke import OCC_DukeMTMCreID
from .vehicleid import VehicleID
from .veri import VeRi

from .sampler_ddp import RandomIdentitySampler_DDP
import torch.distributed as dist


# 数据集类的工厂字典
# 键是数据集名称，值是对应的数据集类或者构造函数
__factory = {
    'market1501': Market1501,
    'dukemtmc': DukeMTMCreID,
    'msmt17': MSMT17,
    'occ_duke': OCC_DukeMTMCreID,
    'veri': VeRi,
    'VehicleID': VehicleID
}

# 将一个 batch 的数据进行整理和处理，以便于模型训练。它将从数据集中提取的元素（如图像、标签等）进行打包，并转换为适合模型输入的格式。
def train_collate_fn(batch):
    """
    # collate_fn这个函数的输入就是一个list，list的长度是一个batch size，list中的每个元素都是__getitem__得到的结果
    """
    # zip(*batch) 将 batch 中的每个元素按位置进行分组。假设 batch 中的每个元素是 (img, pid, camid, viewid, _)，zip(*batch) 将返回五个元组
    imgs, pids, camids, viewids , _ = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    viewids = torch.tensor(viewids, dtype=torch.int64)
    camids = torch.tensor(camids, dtype=torch.int64)
    # 将 pids、camids 和 viewids 转换为 PyTorch 的 int64 类型张量。这是因为模型训练中需要这些标签为张量形式，便于计算和操作。
    return torch.stack(imgs, dim=0), pids, camids, viewids,
    # torch.stack(imgs, dim=0) 将 imgs 列表中的所有图像张量沿着新的维度（这里是第0维）进行堆叠，形成一个 batch 的图像张量。
    # 最终返回：
    # torch.stack(imgs, dim=0)：一个包含所有图像的张量，形状为 (batch_size, C, H, W)。
    # torch.tensor(pids, dtype=torch.int64)：一个包含所有行人ID的张量，形状为 (batch_size,)。
    # torch.tensor(camids, dtype=torch.int64)：一个包含所有摄像机ID的张量，形状为 (batch_size,)。
    # torch.tensor(viewids, dtype=torch.int64)：一个包含所有视角ID的张量，形状为 (batch_size,)。

def val_collate_fn(batch):
    # 整理和处理验证集中的一个batch
    imgs, pids, camids, viewids, img_paths = zip(*batch) # 解包
    viewids = torch.tensor(viewids, dtype=torch.int64) # 将视角ID和摄像机ID转换为张量
    camids_batch = torch.tensor(camids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids, camids, camids_batch, viewids, img_paths

def make_dataloader(cfg):
    # 创建数据加载器，数据预处理与数据增强
    # 训练集
    train_transforms = T.Compose([
            T.Resize(cfg.INPUT.SIZE_TRAIN, interpolation=3),
            T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
            T.Pad(cfg.INPUT.PADDING),
            T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
            T.ToTensor(),
            T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
            RandomErasing(probability=cfg.INPUT.RE_PROB, mode='pixel', max_count=1, device='cpu'),
            # RandomErasing(probability=cfg.INPUT.RE_PROB, mean=cfg.INPUT.PIXEL_MEAN) # 随机擦除
        ])
    # 验证集
    val_transforms = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TEST),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    ])

    num_workers = cfg.DATALOADER.NUM_WORKERS # 数据加载的工作线程数量

    # 根据配置中的数据集名称实例化相应的数据集类
    dataset = __factory[cfg.DATASETS.NAMES](root=cfg.DATASETS.ROOT_DIR)
    # cfg中包含了配置内容，比如数据集的名称与根目录是什么，配置文件在 config/__init__.py 中
    # 所以在默认数据集下 实际执行的是 dataset = Market1501(root='../data')

    # 创建训练集和验证集的ImageDataset实例
    train_set = ImageDataset(dataset.train, train_transforms)
    train_set_normal = ImageDataset(dataset.train, val_transforms)
    num_classes = dataset.num_train_pids # 行人类别数量
    cam_num = dataset.num_train_cams # 摄像机数量
    view_num = dataset.num_train_vids # 视角数量

    # 根据配置选择数据加载的采样器
    if 'triplet' in cfg.DATALOADER.SAMPLER:
        if cfg.MODEL.DIST_TRAIN: # 分布式训练配置
            print('DIST_TRAIN START')
            mini_batch_size = cfg.SOLVER.IMS_PER_BATCH // dist.get_world_size()
            data_sampler = RandomIdentitySampler_DDP(dataset.train, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE)
            batch_sampler = torch.utils.data.sampler.BatchSampler(data_sampler, mini_batch_size, True)
            train_loader = torch.utils.data.DataLoader(
                train_set,
                num_workers=num_workers,
                batch_sampler=batch_sampler,
                collate_fn=train_collate_fn,
                pin_memory=True,
            )
        else:
            train_loader = DataLoader(
                train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH,
                sampler=RandomIdentitySampler(dataset.train, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE),
                num_workers=num_workers, collate_fn=train_collate_fn
            )
    elif cfg.DATALOADER.SAMPLER == 'softmax':
        print('using softmax sampler')
        train_loader = DataLoader(
            train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH, shuffle=True, num_workers=num_workers,
            collate_fn=train_collate_fn
        )
    else:
        print('unsupported sampler! expected softmax or triplet but got {}'.format(cfg.SAMPLER))

    val_set = ImageDataset(dataset.query + dataset.gallery, val_transforms)

    # 验证机数据加载器
    val_loader = DataLoader(
        val_set, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        collate_fn=val_collate_fn
    )
    # 标准化训练集加载器
    train_loader_normal = DataLoader(
        train_set_normal, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        collate_fn=val_collate_fn
    )
    # 训练集加载器，标准化的训练集加载器，验证集加载器，查询集大小，类别数，摄像机数，视角数
    return train_loader, train_loader_normal, val_loader, len(dataset.query), num_classes, cam_num, view_num
