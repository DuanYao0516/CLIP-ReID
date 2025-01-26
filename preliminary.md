## 工具与配置
1. 什么是 YACS？
一种参数配置工具，本质上我并未发现与YAML配置文件的层级结构有所不同，可能我的理解不够深入，快速入门请参考 https://blog.csdn.net/li1784506/article/details/139831426?spm=1001.2101.3001.6650.1&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7Ebaidujs_utm_term%7ECtr-1-139831426-blog-139755195.235%5Ev43%5Epc_blog_bottom_relevance_base7&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7Ebaidujs_utm_term%7ECtr-1-139831426-blog-139755195.235%5Ev43%5Epc_blog_bottom_relevance_base7&utm_relevant_index=2
2. 什么是 timm？
是一个个人开发的Pytorch库，用于提供各种预训练的CNN模型。


## 深度学习相关概念
1. 深度学习中的 fan_in 与 fan_out 是什么？
输入与输出的神经元个数，计算方式请参考 https://blog.csdn.net/wumo1556/article/details/86583946
2. 什么是 SIE？
是 Trans REID 引入的一个模块，the side information embeddings（SIE）。通过插入可学习的 embeddings 来合并这些非视觉线索，以减轻对相机/视角变化的特征偏差。辅助信息嵌入，比如不同的摄像机有不同的参数，将摄像机ID作为一个编码嵌入学习，提升模型性能。
3. neck_feat 是什么？
“neck_feat” 通常指的是网络结构中的 “neck” 部分所提取的特征。在深度学习模型中，尤其是一些用于分类或特征提取的模型（如基于卷积神经网络的模型），通常可以分为 backbone（主干网络）、neck（颈部网络）和 head（头部网络）三个部分。Neck：位于 backbone 和 head 之间，其作用是对 backbone 提取的特征进行进一步的处理和调整，例如进行特征融合、维度变换等操作，以增强特征的表达能力。
4. JIT archive是什么？为什么要加载？
“JIT archive” 指的是使用 PyTorch 的 Just-In-Time (JIT) 编译器将 PyTorch 模型编译后保存的文件。PyTorch JIT 是一种将 Python 代码和 PyTorch 模型转换为中间表示（IR）的工具，这个中间表示可以在不依赖 Python 解释器的情况下进行高效的执行。JIT archive 文件通常具有 .pt 或 .pth 扩展名。
5. 什么是数据集类的工厂字典？
6. 什么是 kaiming 初始化方法？

