from deepvac.syszux_config import *
# DDP
config.dist_url = 'tcp://172.16.90.55:27030'
config.world_size = 1

config.disable_git = True
config.workers = 3
config.device = 'cuda'
config.epoch_num = 200
config.lr = 0.001
config.lr_step = [50, 100]
config.lr_factor = 0.1
config.save_num = 1
config.log_every = 10
config.momentum = 0.99
config.weight_decay=5e-4
config.nesterov = False
config.drop_last = True
config.pin_memory = True

config.output_dir = 'output'

# train
config.train.data_dir = 'your train images path'
config.train.gt_dir = 'your train txts path'
config.train.batch_size = 12
config.train.shuffle = True
config.train.img_size = 640
config.train.is_transform = True
config.train.arch = 'resnet18'

# val
config.val.data_dir = 'your val images path'
config.val.gt_dir = 'your val txts path'
config.val.batch_size = 1
config.val.shuffle = False
config.val.img_size = 640
config.val.is_transform = True

#test
#config.model_path = 'your model path'
config.test.fileline_data_path_prefix = 'your test image path'
config.test.fileline_path = 'your test txt path'
config.test.batch_size = 1
config.test.shuffle = False
config.test.arch = 'resnet18'
config.test.long_size = 1280
config.test.use_fileline = True
