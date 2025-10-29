# HDA-Net

## 1. 环境配置

在目录下创建并激活：

```
conda env create -f env.yml
conda activate HDA
```

## 2. 数据预处理

使用文件夹preprocess中的process_lists.py文件可以将data目录下的两个数据组（train数据组包含131组切割后的标签以及原始的图片，test数据组包含70组原始的图片），经过预处理，将train训练组的同组数据拼接，并且将两组都从nii文件转变为h5文件。

随后使用da_batch_lists.py可以比较train组与test组的部分区别，包括brrrr（这里到时候用gpt）

## 3. 训练开始

使用

