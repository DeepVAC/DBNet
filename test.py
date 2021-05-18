from deepvac import LOG, Deepvac
from modules.utils import SegDetectorRepresenter
import torch
import time
import cv2
import os

class DeepvacDBTest(Deepvac):
    def __init__(self, deepvac_config):
        super(DeepvacDBTest,self).__init__(deepvac_config)
        self.post_process = SegDetectorRepresenter()
        self.is_output_polygon = True if self.config.is_output_polygon is None else self.config.is_output_polygon

    def save_image(self, img, idx):
        cv2.imwrite(os.path.join(self.config.output_dir ,str(idx).zfill(3)+'.jpg'), img)

    def report(self):
        self.config.net.eval()
        for index, (org_img, img) in enumerate(self.config.test_loader):
            LOG.logI('progress: %d / %d'%(index, len(self.config.test_loader)))
            org_img = org_img.numpy().astype('uint8')[0]

            img = img.to(self.config.device)
            start_time = time.time()
            preds = self.config.net(img)
            if str(self.config.device).__contains__('cuda'):
                torch.cuda.synchronize(self.config.device)
            print(time.time()-start_time)

            box_list, score_list = self.post_process({'shape': [(org_img.shape[0], org_img.shape[1])]}, preds, is_output_polygon=self.is_output_polygon)
            box_list, score_list = box_list[0], score_list[0]
            if len(box_list) <=0:
                self.save_image(org_img, index)
                continue
                
            if self.is_output_polygon:
                idx = [x.sum() > 0 for x in box_list]
                box_list = [box_list[i] for i, v in enumerate(idx) if v]
                score_list = [score_list[i] for i, v in enumerate(idx) if v]
            else:
                idx = box_list.reshape(box_list.shape[0], -1).sum(axis=1) > 0
                box_list, score_list = box_list[idx], score_list[idx]

            for point in box_list:
                point = point.astype(int)
                cv2.polylines(org_img, [point], True, (0, 255, 0), 2)
            self.save_image(org_img, index)


    def process(self, input_tensor=None):
        self.report()


if __name__ == '__main__':
    from config import config as deepvac_config
    db = DeepvacDBTest(deepvac_config)
    input_tensor = torch.rand(1,3,640,640)
    db(input_tensor)
