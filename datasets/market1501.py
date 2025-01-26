# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import glob
import re
import os.path as osp
from .bases import BaseImageDataset
from collections import defaultdict
import pickle
class Market1501(BaseImageDataset):
    """
    Market1501
    Reference:
    Zheng et al. Scalable Person Re-identification: A Benchmark. ICCV 2015.
    URL: http://www.liangzheng.org/Project/project_reid.html

    - 1501个身份
    - 6个摄像头
    - 32668张图片
    - DPM检测器代替手工框出行人
    - 500K张干扰图片
    - 每一个身份有多个query
    - 每一个query平均对应14.8个gallery

    Dataset statistics:
    # identities: 1501 (+1 for background)
    # images: 12936 (train) + 3368 (query) + 15913 (gallery)
    """
    # 在这里设置数据集名称
    dataset_dir = 'Market-1501-v15.09.15'

    def __init__(self, root='', verbose=True, pid_begin = 0, **kwargs):
        super(Market1501, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'bounding_box_train')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'bounding_box_test')

        # 检查数据集文件是否存在
        self._check_before_run()
        self.pid_begin = pid_begin
        # 分别处理训练集、查询集、图库集
        train = self._process_dir(self.train_dir, relabel=True)
        query = self._process_dir(self.query_dir, relabel=False)
        gallery = self._process_dir(self.gallery_dir, relabel=False)

        if verbose:
            print("=> Market1501 loaded")
            self.print_dataset_statistics(train, query, gallery)

        # 将处理后的数据集保存在类属性中
        self.train = train
        self.query = query
        self.gallery = gallery
        # 获取数据集统计信息
        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = self.get_imagedata_info(self.gallery)

    def _check_before_run(self):
        # 检查数据集文件是否存在
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def _process_dir(self, dir_path, relabel=False):
        # 获取指定目录下 所有 .jpg 格式的图像文件路径
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        # 正则表达式，用于解析图像文件名，提取行人ID与摄像机ID
        pattern = re.compile(r'([-\d]+)_c(\d)')

        pid_container = set()
        for img_path in sorted(img_paths): #  遍历所有图像路径
            # 从文件名中提取行人ID与摄像机ID，忽略无效图像
            pid, _ = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            pid_container.add(pid)
        # 如果需要重新标注，则创建一个从行人ID到标签的映射字典
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        # pid_container: {2, 7, 10, 11, 12, 20, 22, 23......}
        # pid2label: {2: 0, 7:1, 10:2, 11:3, 12:4, 20:5, 22:6, 23:7......}

        # 创建一个列表，存储处理后的数据集
        dataset = []
        """
        '0002_c1s1_000451_03.jpg'
        0002 是行人 ID，Market 1501 有 1501 个行人，故行人 ID 范围为 0001-1501
        c1 是摄像头编号(camera 1)，表明图片采集自第1个摄像头，一共有 6 个摄像头
        s1 是视频的第一个片段(sequece1)，一个视频包含若干个片段
        000451 是视频的第 451 帧图片，表明行人出现在该帧图片中
        03 代表第 451 帧图片上的第三个检测框，DPM 检测器可能在一帧图片上生成多个检测框
        """
        for img_path in sorted(img_paths):
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            assert 0 <= pid <= 1501  # pid == 0 means background
            assert 1 <= camid <= 6
            camid -= 1  # 摄像机 index starts from 0
            if relabel: pid = pid2label[pid] # 如果需要重新标注，使用新的行人ID
            # 将图像路径、行人ID、摄像机ID和初始标签（0）添加到数据集中
            dataset.append((img_path, self.pid_begin + pid, camid, 0))
        return dataset