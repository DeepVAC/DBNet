import torch
from deepvac import new, AttrDict
from deepvac.datasets import CocoCVContoursDataset

config = new(None)

config.datasets.CocoCVContoursDataset = AttrDict()
config.datasets.CocoCVContoursDataset.auto_detect_subdir_with_basenum = 0

sample_path_prefix_list = ['your sample path prefix list']
target_path_list = ['your json file path list']
config.output_label_dir = 'your output label dir'
config.output_image_dir = 'your output image dir'
config.show = True

config.test_loader_list = []
for i in range(len(sample_path_prefix_list)):
    test_dataset = CocoCVContoursDataset(config, sample_path_prefix_list[i], target_path_list[i])
    config.test_loader_list.append(torch.utils.data.DataLoader(test_dataset, batch_size=1, pin_memory=False))
