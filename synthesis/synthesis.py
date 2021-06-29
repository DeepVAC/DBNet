import os
import numpy as np
import cv2
from deepvac import LOG

class Synthesis(object):
    def __init__(self, deepvac_config):
        self.config = deepvac_config
        self.auditConfig()
    
    def auditConfig(self):
        if not os.path.exists(self.config.output_label_dir):
            os.makedirs(self.config.output_label_dir)
        
        if self.config.show and not os.path.exists(self.config.output_image_dir):
            os.makedirs(self.config.output_image_dir)
    
    def synthesis(self):
        for i, (img, bbox_info, file_path, show_img) in enumerate(self.test_loader):
            file_path = file_path[0]
            file_name = file_path.split('/')[-1]
            txt_name = 'gt_{}'.format(file_name.replace('jpg', 'txt'))
            fw = open(os.path.join(self.config.output_label_dir, txt_name), 'w')
            for s in bbox_info:
                fw.write('{}\n'.format(s[0]))
            fw.close()

            if self.config.show:
                cv2.imwrite(os.path.join(self.config.output_image_dir, 'bbox_{}'.format(file_name)), show_img.numpy()[0])

    def __call__(self):
        for i, test_loader in enumerate(self.config.test_loader_list):
            LOG.logI('Process dataset {}'.format(i+1))
            self.test_loader = test_loader
            self.synthesis()

if __name__ == "__main__":
    from config import config as deepvac_config

    synthesis = Synthesis(deepvac_config)
    synthesis()