import os
import numpy as np
import cv2

import torch
from deepvac import LOG
from deepvac.datasets import DatasetBase

class CocoCVOcrDataset(DatasetBase):
    def __init__(self, deepvac_config, sample_path_prefix, target_path):
        super(CocoCVOcrDataset, self).__init__(deepvac_config)
        try:
            from pycocotools.coco import COCO
        except:
            raise Exception("pycocotools module not found, you should try 'pip3 install pycocotools' first!")
        self.sample_path_prefix = sample_path_prefix
        self.coco = COCO(target_path)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.cats = list(sorted(self.coco.cats.keys()))
        LOG.logI("Notice: 0 will be treated as background in {}!!!".format(self.name()))

    def auditConfig(self):
        self.auto_detect_subdir_with_basenum = self.addUserConfig('auto_detect_subdir_with_basenum', self.config.auto_detect_subdir_with_basenum, 0)
        LOG.logI("You set auto_detect_subdir_with_basenum to {}".format(self.auto_detect_subdir_with_basenum))

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        id = self.ids[index]
        sample, bbox_info, file_path = self._getSample(id)
        sample, bbox_info, file_path = self.compose((sample, bbox_info, os.path.join(self.sample_path_prefix, file_path)))
        return sample, bbox_info, file_path

    def updatePath(self, id, file_path):
        full_file_path = self.coco.loadImgs(id)[0]["path"]
        path_list = full_file_path.split('/')
        path_list_num = len(path_list)

        if path_list_num == self.auto_detect_subdir_with_basenum:
            withsub_file_path = file_path
        elif path_list_num == self.auto_detect_subdir_with_basenum + 1:
            withsub_file_path = path_list[-2] + '/' + file_path
            file_path = path_list[-2] + '_' + file_path
        else:
            LOG.logE("path list has {} fields, which should be {} or {}".format(path_list_num, self.auto_detect_subdir_with_basenum, self.auto_detect_subdir_with_basenum+1), exit=True)

        return withsub_file_path, file_path

    def _bbox_generator(self, img, anns_list):
        bbox_info = []
        for idx, single in enumerate(anns_list):
            s = ''
            if single['isbbox']:
                x, y, w, h = single['bbox']
                x, y, w, h = int(x), int(y), int(w), int(h)
                s = '{},{},{},{},{},{},{},{},ok'.format(x, y, x+w, y, x+w, y+h, x, y+h)
            else:
                seg = single['segmentation']
                pts = np.array(seg[0]).reshape((-1,2)).astype(np.int32)
                for pt in pts.reshape((-1)):
                    s += '{},'.format(pt)
                s += 'ok'
            bbox_info.append(s)
        return bbox_info

    def _getSample(self, id: int):
        # img
        file_path = self.coco.loadImgs(id)[0]["file_name"]
        withsub_file_path = file_path
        if self.auto_detect_subdir_with_basenum > 0:
            withsub_file_path, file_path = self.updatePath(id, file_path)

        img = cv2.imread(os.path.join(self.sample_path_prefix, file_path), 1)
        assert img is not None, "Image {} not found in {} !".format(file_path, self.sample_path_prefix)
        # anno
        anns = self.coco.loadAnns(self.coco.getAnnIds(id))

        bbox_info = self._bbox_generator(img, anns)

        # return target you want
        return img, bbox_info, file_path