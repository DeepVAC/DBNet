# DBNet
DeepVAC-compliant DBNet implementation.

# 简介
本项目实现了符合DeepVAC规范的OCR检测模型DBNet

**项目依赖**

- deepvac >= 0.5.6
- pytorch >= 1.8.0
- torchvision >= 0.7.0
- opencv-python
- numpy
- pyclipper
- shapely
- pillow

# 如何运行本项目

## 1. 阅读[DeepVAC规范](https://github.com/DeepVAC/deepvac)
可以粗略阅读，建立起第一印象

## 2. 准备运行环境
可以使用DeepVAC规范指定的[Docker镜像](https://github.com/DeepVAC/deepvac#2-%E7%8E%AF%E5%A2%83%E5%87%86%E5%A4%87)

## 3. 准备数据集
- 获取文本检测数据集
  CTW1500格式的数据集，CTW1500下载地址:

  [ch4_training_images.zip](https://rrc.cvc.uab.es/downloads/ch4_training_images.zip)

  [ch4_training_localization_transcription_gt.zip](https://rrc.cvc.uab.es/downloads/ch4_training_localization_transcription_gt.zip)

  [ch4_test_images.zip](https://rrc.cvc.uab.es/downloads/ch4_test_images.zip)

  [Challenge4_Test_Task1_GT.zip](https://rrc.cvc.uab.es/downloads/Challenge4_Test_Task1_GT.zip)

- 数据集配置
  在config.py文件中作如下配置:

```python
# line 47-48
sample_path = <your train image path>
label_path = <your train gt path>
# line 66-67
sample_path = <your val image path>
label_path = <your val gt path>
```

## 5. 模型相关配置

- DB backbone配置

```python
# line 32, 目前支持resnet18，mv3large
arch = "resnet18"
```

## 4. 训练相关配置

- dataloader相关配置

```python
# line 49-63
is_transform = True        # 是否做数据增强
img_size = 640             # 训练图片大小(img_size, img_size)
config.datasets.DBTrainDataset = AttrDict()
config.datasets.DBTrainDataset.shrink_ratio = 0.4
config.datasets.DBTrainDataset.thresh_min = 0.3
config.datasets.DBTrainDataset.thresh_max = 0.7
config.core.train_dataset = DBTrainDataset(config, sample_path, label_path, is_transform, img_size)
config.core.train_loader = torch.utils.data.DataLoader(
  dataset = config.core.train_dataset,
  batch_size = 12,
  shuffle = True,
  num_workers = 4,
  pin_memory = True,
  sampler = None
)
```

## 5. 训练

```
python3 train.py
```

## 6. 测试

- 测试相关配置

```python
# line 80-90
config.core.model_path = <your model path>            # 加载模型路径
config.core.is_output_polygon = True                  # 输出是否为多边形模型
sample_path = <your test image path>                     # 测试图片路径
config.core.test_dataset = DBTestDataset(config, sample_path, long_size = 1280)
config.core.test_loader = torch.utils.data.DataLoader(
  dataset = config.core.test_dataset,
  batch_size = 1,
  shuffle = False,
  num_workers = 0,
  pin_memory = True
)
```

- 运行测试脚本:

```bash
python3 test.py
```

## 7. 使用torchscript模型

  如果训练过程中未开启config.cast.script_model_dir开关，可以在测试过程中转化torchscript模型

  - 转换torchscript模型(.pt)

  ```python
  config.cast.script_model_dir = "output/script.pt"
  ```

  按照步骤6完成测试，torchscript模型会保存至config.cast.script_model_dir指定位置

  - 加载torchscript模型

  ```python
  config.core.jit_model_path = <torchscript-model-path>
  ```
  然后按照步骤6测试，会读取script_model

## 更多功能

  如果要在本项目中开启如下功能:

  - 预训练模型加载
  - checkpoint加载
  - 使用tensorboard
  - 启用TorchScript
  - 转换ONNX
  - 转换NCNN
  - 转换CoreML
  - 开启量化
  - 开启自动混合精度训练
  - 采用ema策略(config.ema)
  - 采用梯度积攒到一定数量再进行反向更新梯度策略(config.nominal_batch_factor)

  请参考[DeepVAC](https://github.com/DeepVAC/deepvac)
