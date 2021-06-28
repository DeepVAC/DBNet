import torch
from deepvac import new, AttrDict
from deepvac.datasets import CocoCVOcrDataset

config = new(None)

config.datasets.CocoCVOcrDataset = AttrDict()
config.datasets.CocoCVOcrDataset.auto_detect_subdir_with_basenum = 0

config.sample_path_prefix = 'your sample path prefix'
config.target_path = 'your json file path'
config.output_dir = 'your output dir'

config.test_dataset = CocoCVOcrDataset(config, config.sample_path_prefix, config.target_path)
config.test_loader = torch.utils.data.DataLoader(config.test_dataset, batch_size=1, pin_memory=False)
