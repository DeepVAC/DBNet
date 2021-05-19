import torch
import torch.optim as optim

from deepvac import config

from data.dataloader import DBTrainDataset, DBTestDataset
from modules.model_db import Resnet18DB, Mobilenetv3LargeDB
from modules.loss import DBLoss
## ------------------ common ------------------
config.core.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
config.core.output_dir = 'output'
config.core.log_every = 100
config.core.disable_git = True
config.core.model_reinterpret_cast = True
config.core.cast_state_dict_strict = False

## -------------------- training ------------------
## train runtime
config.core.epoch_num = 200
config.core.save_num = 1

## -------------------- tensorboard ------------------
#config.core.tensorboard_port = "6007"
#config.core.tensorboard_ip = None

## -------------------- script and quantize ------------------
#config.core.trace_model_dir = "./trace.pt"
#config.core.static_quantize_dir = "./script.sq"
#config.core.dynamic_quantize_dir = "./quantize.sq"

## -------------------- net and criterion ------------------
arch = "resnet18"
if arch == "resnet18":
    config.core.net = Resnet18DB()
elif arch == "mv3large":
    config.core.net = Mobilenetv3LargeDB()
else:
    raise Exception("Architecture {} is not supported!".format(arch))
config.core.criterion = DBLoss(config)

## -------------------- optimizer and scheduler ------------------
config.core.optimizer = torch.optim.Adam(config.core.net.parameters(), 5e-4, (0.9, 0.999), eps=1e-08, weight_decay=5e-4)

lambda_lr = lambda epoch: round ((1 - epoch/config.core.epoch_num) ** 0.9, 8)
config.core.scheduler = optim.lr_scheduler.LambdaLR(config.core.optimizer, lr_lambda=lambda_lr)

## -------------------- loader ------------------
data_dir = 'your train image dir'
gt_dir = 'your train labels dir'
is_transform = True
img_size = 640
config.core.shrink_ratio = 0.4
config.core.thresh_min = 0.3
config.core.thresh_max = 0.7
config.core.train_dataset = DBTrainDataset(config, data_dir, gt_dir, is_transform, img_size)
config.core.train_loader = torch.utils.data.DataLoader(
    dataset = config.core.train_dataset,
    batch_size = 12,
    shuffle = True,
    num_workers = 4,
    pin_memory = True,
    sampler = None
)

## -------------------- val ------------------
data_dir = 'your val image dir'
gt_dir = 'your val labels dir'
is_transform = True
img_size = 640
config.core.val_dataset = DBTrainDataset(config, data_dir, gt_dir, is_transform, img_size)
config.core.val_loader = torch.utils.data.DataLoader(
    dataset = config.core.val_dataset,
    batch_size = 1,
    shuffle = False,
    num_workers = 0,
    pin_memory = True
)

## -------------------- test ------------------
config.core.model_path = "your test model dir / pretrained weights"
config.core.is_output_polygon = True
data_dir = 'your test image dir'
config.core.test_dataset = DBTestDataset(config, data_dir, long_size = 1280)
config.core.test_loader = torch.utils.data.DataLoader(
    dataset = config.core.test_dataset,
    batch_size = 1,
    shuffle = False,
    num_workers = 0,
    pin_memory = True
)

## ------------------------- DDP ------------------
config.dist_url = 'tcp://172.16.90.55:27030'
config.world_size = 1
