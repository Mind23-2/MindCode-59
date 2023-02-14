# 目录

<!-- TOC -->

- [目录](#目录)
- [MTCNN描述](#mtcnn描述)
- [模型架构](#模型架构)
- [数据集](#数据集)
- [特性](#特性)
    - [混合精度](#混合精度)
- [环境要求](#环境要求)
- [快速入门](#快速入门)
- [脚本说明](#脚本说明)
    - [脚本及样例代码](#脚本及样例代码)
    - [脚本参数](#脚本参数)
    - [训练过程](#训练过程)
        - [训练](#训练)
        - [分布式训练](#分布式训练)
    - [评估过程](#评估过程)
        - [评估](#评估)
- [模型描述](#模型描述)
    - [性能](#性能)
        - [评估性能](#评估性能)
            - [FDDB上的MTCNN](#FDDB上的MTCNN)
        - [推理性能](#推理性能)
            - [FDDB上的MTCNN](#FDDB上的MTCNN)
    - [使用流程](#使用流程)
        - [推理](#推理)
        - [迁移学习](#迁移学习)
- [随机情况说明](#随机情况说明)
- [ModelZoo主页](#modelzoo主页)

<!-- /TOC -->

# MTCNN描述

MTCNN(Multi-task Cascaded Convolutional Networks)是 一种多任务级联卷积神经网络，用以同时处理人脸检测和人脸关键点定位问题。作者认为人脸检测和人脸关键点检测两个任务之间往往存在着潜在的联系，然而大多数方法都未将两个任务有效的结合起来，MTCNN充分利用两任务之间潜在的联系，将人脸检测和人脸关键点检测同时进行，可以实现人脸检测和5个特征点的标定。

[论文](https://kpzhang93.github.io/MTCNN_face_detection_alignment/)：[1] Zhang K ,  Zhang Z ,  Li Z , et al. Joint Face Detection and Alignment Using Multitask Cascaded Convolutional Networks[J]. IEEE Signal Processing Letters, 2016, 23(10):1499-1503.

# 模型架构

MTCNN为了解决人脸识别的两阶段问题，提出三个级联的多任务卷积神经网络（Proposal Network (P-Net)、Refine Network (R-Net)、Output Network (O-Net)，每个多任务卷积神经网络均有三个学习任务，分别是人脸分类、边框回归和关键点定位。每一级的输出作为下一级的输入。

# 数据集

使用的数据集：WIDER FACE

- 数据集大小：包含32203图像，393,703个标注人脸
    - 训练集：158,989个标注人脸位于训练集
    - 测试集：39,496个位于验证集
- 数据格式：RGB
    - 注：

# 特性

## 混合精度

采用[混合精度](https://www.mindspore.cn/tutorial/training/zh-CN/master/advanced_use/enable_mixed_precision.html)的训练方法使用支持单精度和半精度数据来提高深度学习神经网络的训练速度，同时保持单精度训练所能达到的网络精度。混合精度训练提高计算速度、减少内存使用的同时，支持在特定硬件上训练更大的模型或实现更大批次的训练。
以FP16算子为例，如果输入数据类型为FP32，MindSpore后台会自动降低精度来处理数据。用户可打开INFO日志，搜索“reduce precision”查看精度降低的算子。

# 环境要求

- 硬件（Ascend/GPU）
    - 使用Ascend或GPU处理器来搭建硬件环境。
- 框架
    - [MindSpore](https://www.mindspore.cn/install/en)
- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorial/training/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/doc/api_python/en/master/index.html)

# 快速入门

通过官方网站安装MindSpore后，您可以按照如下步骤进行训练和评估：

- Ascend处理器环境运行

  ```python
  # 运行训练示例
  python3 train.py > train.log 2>&1 & --dataset_path /path/dataset --pre_ckpt_path /path/pretrained_path --isModelArts False --distribute False
  OR
  bash run_single_train.sh /path/dataset 0 /path/pretrained_path


  # 运行分布式训练示例
  python3 train.py > train.log 2>&1 & --dataset_path /path/dataset --pre_ckpt_path /path/pretrained_path --isModelArts False
  OR
  bash run_distribute_train.sh /path/dataset /path/pretrain_path 8

  # 运行评估示例
  python3 eval.py > eval.log 2>&1 & --dataset_path /path/dataset --ckpt_path /path/ckpt
  或
  bash run_eval.sh /path/dataset 0 /path/ckpt
  ```

  对于分布式训练，需要提前创建JSON格式的hccl配置文件。

  请遵循以下链接中的说明：

 <https://gitee.com/mindspore/mindspore/tree/master/model_zoo/utils/hccl_tools.>

- 默认使用ImageNet2012数据集。您也可以将`$dataset_type`传入脚本，以便选择其他数据集。如需查看更多详情，请参考指定脚本。

# 脚本说明

## 脚本及样例代码

```bash
├── model_zoo
    ├── README.md                          // 所有模型相关说明
    ├── MTCNN
        ├── README.md                    // hardnet相关说明
        ├── scripts
        │   ├──run_single_train.sh             // 单卡到Ascend的shell脚本
        │   ├──run_distribute_train.sh             // 分布式到Ascend的shell脚本
        │   ├──run_eval.sh              // Ascend评估的shell脚本
        ├── src
        │   ├──dataset.py             // 创建数据集
        │   ├──mtcnnt.py          //  mtcnn架构
        │   ├──image_tools.py            // 图片处理工具函数
        │   ├──imagedb.py            // 图片处理构造
        |   ├──config.py                 //参数配置
        |   ├──data_utils.py         //数据处理工具相关
        |   ├──img_utils.py         //图片处理工具相关
        │   ├──detect.py            // 人脸识别
        ├── train_pnet.py               // 训练P网络
        ├── train_rnet.py               // 训练R网络
        ├── train_onet.py               // 训练O网络
        
```


### 迁移学习

待补充

# 随机情况说明

在dataset.py中，我们设置了“create_dataset”函数内的种子，同时还使用了train.py中的随机种子。

# ModelZoo主页  

 请浏览官网[主页](https://gitee.com/mindspore/mindspore/tree/master/model_zoo)。
