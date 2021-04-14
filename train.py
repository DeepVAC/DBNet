import sys
from deepvac import DeepvacTrain, LOG
import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
from torch.autograd import Variable

from modules.model_db import Resnet18DB, Mobilenetv3LargeDB
from modules.utils import cal_text_score
from modules.loss import DBLoss
from modules.metrics import runningScore
from data.data_loader import DBTrainDataset

import cv2
import os
import time
import numpy as np

class DeepvacDB(DeepvacTrain):
    def __init__(self, deepvac_config):
        super(DeepvacDB,self).__init__(deepvac_config)

    def initNetWithCode(self):
        if self.conf.train.arch == "resnet18":
            self.net = Resnet18DB()
        elif self.conf.train.arch == "mv3":
            self.net = Mobilenetv3LargeDB()

        self.net.to(self.device)

    def initOptimizer(self):
        self.initAdamOptimizer()

    def initCriterion(self):
        self.criterion = DBLoss()

    def initTrainLoader(self):
        self.train_dataset = DBTrainDataset(self.conf.train)
        self.train_loader = DataLoader(
            dataset = self.train_dataset,
            batch_size=self.conf.train.batch_size,
            shuffle=self.conf.train.shuffle,
            num_workers=self.conf.workers,
            drop_last=self.conf.drop_last,
            pin_memory=self.conf.pin_memory,
        )

    def initValLoader(self):
        self.val_dataset = DBTrainDataset(self.conf.val)
        self.val_loader = DataLoader(
            dataset = self.val_dataset,
            batch_size=self.conf.val.batch_size,
            shuffle=self.conf.val.shuffle,
            num_workers=self.conf.workers,
            drop_last=self.conf.drop_last,
            pin_memory=self.conf.pin_memory,
        )

    def initTestLoader(self):
        pass

    def preIter(self):
        pass

    def earlyIter(self):
        self.sample = self.sample.to(self.device)
        if not self.is_train:
            return

        try:
            self.addGraph(self.sample)
        except:
            LOG.logW("Tensorboard addGraph failed. You network foward may have more than one parameters?")
            LOG.logW("Seems you need reimplement preIter function.")

    def doForward(self):
        self.outputs = self.net(self.sample).to('cpu')

    def doLoss(self):
        self.loss = self.criterion(self.outputs, self.target)['loss']

    def postIter(self):
        if self.is_train:
            return

        self.score_text = cal_text_score(self.outputs[:, 0, :, :], self.target[0], self.target[1], self.running_metric_text)

    def preEpoch(self):
        if self.is_train:
            return 
        self.running_metric_text = runningScore(2)

    def postEpoch(self):
        if self.is_train:
            return

        self.accuracy = self.score_text['Mean Acc']
        LOG.logI('Test accuray: {:.4f}'.format(self.accuracy))

    def processAccept(self):
        return

if __name__ == '__main__':
    from config import config as deepvac_config
    DB = DeepvacDB(deepvac_config)
    DB()
