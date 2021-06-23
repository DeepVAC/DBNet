import torch
import torch.optim as optim

from deepvac import config, AttrDict, new

from data.dataloader import DBTrainDataset, DBTestDataset
from modules.model_db import Resnet18DB, Mobilenetv3LargeDB
from modules.loss import DBLoss

config = new('DBNetTrain')
## ------------------ common ------------------
config.core.DBNetTrain.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
config.core.DBNetTrain.output_dir = 'output'
config.core.DBNetTrain.log_every = 100
config.core.DBNetTrain.disable_git = True
config.core.DBNetTrain.model_reinterpret_cast = True
config.core.DBNetTrain.cast_state_dict_strict = True
# config.core.DBNetTrain.jit_model_path = "./output/script.pt"

## -------------------- training ------------------
## train runtime
config.core.DBNetTrain.epoch_num = 200
config.core.DBNetTrain.save_num = 1

## -------------------- tensorboard ------------------
#config.core.DBNetTrain.tensorboard_port = "6007"
#config.core.DBNetTrain.tensorboard_ip = None

## -------------------- script and quantize ------------------
config.cast.TraceCast = AttrDict()
config.cast.TraceCast.model_dir = "./script.pt"
# config.cast.TraceCast.static_quantize_dir = "./script.sq"     # unsupported op nn.ConvTranspose2d for now
config.cast.TraceCast.dynamic_quantize_dir = "./quantize.sq"

## -------------------- net and criterion ------------------
config.arch = "resnet18"
if config.arch == "resnet18":
    config.core.DBNetTrain.net = Resnet18DB()
elif config.arch == "mv3large":
    config.core.DBNetTrain.net = Mobilenetv3LargeDB()
else:
    raise Exception("Architecture {} is not supported!".format(config.arch))
config.core.DBNetTrain.criterion = DBLoss(config)

## -------------------- optimizer and scheduler ------------------
config.core.DBNetTrain.optimizer = torch.optim.Adam(config.core.DBNetTrain.net.parameters(), 5e-4, (0.9, 0.999), eps=1e-08, weight_decay=5e-4)

lambda_lr = lambda epoch: round ((1 - epoch/config.core.DBNetTrain.epoch_num) ** 0.9, 8)
config.core.DBNetTrain.scheduler = optim.lr_scheduler.LambdaLR(config.core.DBNetTrain.optimizer, lr_lambda=lambda_lr)

## -------------------- loader ------------------
config.sample_path = 'your train image dir'
config.label_path = 'your train labels dir'
config.is_transform = True
config.img_size = 640
config.datasets.DBTrainDataset = AttrDict()
config.datasets.DBTrainDataset.shrink_ratio = 0.4
config.datasets.DBTrainDataset.thresh_min = 0.3
config.datasets.DBTrainDataset.thresh_max = 0.7
config.core.DBNetTrain.batch_size = 8
config.core.DBNetTrain.num_workers = 4
config.core.DBNetTrain.train_dataset = DBTrainDataset(config, config.sample_path, config.label_path, config.is_transform, config.img_size)
config.core.DBNetTrain.train_loader = torch.utils.data.DataLoader(
    dataset = config.core.DBNetTrain.train_dataset,
    batch_size = config.core.DBNetTrain.batch_size,
    shuffle = True,
    num_workers = config.core.DBNetTrain.num_workers,
    pin_memory = True,
    sampler = None
)

## -------------------- val ------------------
config.sample_path = 'your val image dir'
config.label_path = 'your val labels dir'
config.is_transform = True
config.img_size = 640
config.core.DBNetTrain.val_dataset = DBTrainDataset(config, config.sample_path, config.label_path, config.is_transform, config.img_size)
config.core.DBNetTrain.val_loader = torch.utils.data.DataLoader(
    dataset = config.core.DBNetTrain.val_dataset,
    batch_size = 1,
    shuffle = False,
    num_workers = 0,
    pin_memory = True
)

## -------------------- test ------------------
config.core.DBNetTest = config.core.DBNetTrain.clone()
config.core.DBNetTest.model_path = 'your test model dir / pretrained weights'
# config.core.DBNetTest.jit_model_path = 'your torchscript model path'
config.core.DBNetTest.is_output_polygon = True
config.sample_path = 'your test image dir'
config.core.DBNetTest.test_dataset = DBTestDataset(config, config.sample_path, long_size = 1280)
config.core.DBNetTest.test_loader = torch.utils.data.DataLoader(
    dataset = config.core.DBNetTest.test_dataset,
    batch_size = 1,
    shuffle = False,
    num_workers = 0,
    pin_memory = True
)

## ------------------------- DDP ------------------
config.core.DBNetTrain.dist_url = 'tcp://172.16.90.55:27030'
config.core.DBNetTrain.world_size = 1
