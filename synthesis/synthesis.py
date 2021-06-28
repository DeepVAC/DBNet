import os

def synthesis(deepvac_config):
    for i, (img, bbox_info, file_path) in enumerate(deepvac_config.test_loader):
        file_path = file_path[0]
        file_name = file_path.split('/')[-1]
        txt_name = 'gt_{}'.format(file_name.replace('jpg', 'txt'))
        fw = open(os.path.join(deepvac_config.output_dir, txt_name), 'w')
        for s in bbox_info:
            fw.write('{}\n'.format(s[0]))
        fw.close()

if __name__ == "__main__":
    from config import config as deepvac_config

    synthesis(deepvac_config)