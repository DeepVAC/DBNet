import torch
from deepvac import new, AttrDict
from data.dataloader import CocoCVOcrDataset

config = new(None)

config.datasets.CocoCVOcrDataset = AttrDict()
config.datasets.CocoCVOcrDataset.auto_detect_subdir_with_basenum = 0

config.sample_path_prefix = '/gemfield/hostpv/wangyuhang/gitlab/deepvac-ocr-det/JPEGImages'
config.target_path = '/opt/public/airlock/lihang/zangzu_detect-235.json'
config.output_dir = './gt'

config.test_dataset = CocoCVOcrDataset(config, config.sample_path_prefix, config.target_path)
config.test_loader = torch.utils.data.DataLoader(config.test_dataset, batch_size=1, pin_memory=False)
