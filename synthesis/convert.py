import os
import yaml
from argparse import ArgumentParser
from pycocotools.coco import COCO

import numpy as np
import cv2

def mask_generator(coco, width, height, anns_list, cate2label, id2cate):
    mask_pic = np.zeros((height, width))
    intersection = np.zeros((height, width))
    for single in anns_list:
        print(single)
        if single['isbbox']:
            continue
        mask_single = coco.annToMask(single)
        intersection += mask_single
        mask_pic += mask_single*cate2label[id2cate[single['category_id']]]
    mask_pic = mask_pic.astype(int)
    mask_pic[np.where(intersection>1)] = 0
    return mask_pic

def bbox_generator(anns_list, img_name, output_dir):
    txt_name = 'gt_'+img_name[:-4]+'.txt'
    fw = open(os.path.join(output_dir,txt_name), 'w')
    # bbox_pic = image.copy()
    for idx, single in enumerate(anns_list):
        if single['isbbox']:
            x, y, w, h = single['bbox']
            x, y, w, h = int(x), int(y), int(w), int(h)
            # cv2.rectangle(bbox_pic, (x, y), (x + w, y + h), (0, 0, 255), 2)
            s = str(x)+','+str(y)+','+str(x+w)+','+str(y)+','+str(x+w)+','+str(y+h)+','+str(x)+','+str(y+h)+',ok'
        else:
            seg = single['segmentation']
            #assert len(seg) == 1, 'error'
            if len(seg) != 1:
                print(single['id'])
                print(seg)
            pts = np.array(seg[0]).reshape((-1,2)).astype(np.int32)
            # cv2.polylines(bbox_pic,[pts],True,(0,0,255),2)
            s = ''
            for pt in pts.reshape((-1)):
                s = s+str(pt)
                s += ','
            s += 'ok'
        fw.write(s+'\n')

    fw.close()
    # return bbox_pic

def check(l1, l2):
    for cate in l2:
        if cate not in l1:
            raise Exception('{} in coco but not in yaml'.format(cate))

def main(cfg):
    with open(cfg.cate2label, encoding='utf-8') as f:
        cate2label = yaml.safe_load(f)
    print(cate2label)

    coco = COCO(cfg.coco_json)
    cats = coco.loadCats(coco.getCatIds())
    nms=[cat['name'] for cat in cats]
    print('COCO categories: \n{}\n'.format(' '.join(nms)))
    classes_ids = coco.getCatIds(catNms = nms)
    print(classes_ids)

    id2cate = {idx: cate for idx, cate in zip(classes_ids, nms)}
    print(id2cate)

    print(list(cate2label.keys()))
    print(list(id2cate.values()))

    check(list(cate2label.keys()), list(id2cate.values()))

    imgIds_list = []
    for idx in classes_ids:
        imgidx = coco.getImgIds(catIds=idx)
        imgIds_list += imgidx

    imgIds_list = list(set(imgIds_list))

    image_info_list = coco.loadImgs(imgIds_list)
    output_dir = cfg.output

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for idx, imageinfo in enumerate(image_info_list):
        file_name = imageinfo['file_name'].split('/')[-1]
        print(file_name)
        # image = cv2.imread(os.path.join('JPEGImages',file_name))
        annIds = coco.getAnnIds(imgIds = imageinfo['id'], catIds = classes_ids, iscrowd=None)
        anns_list = coco.loadAnns(annIds)
        bbox_generator(anns_list, file_name, output_dir)
        # bbox_image = bbox_generator(image, anns_list, file_name, output_dir)
        # cv2.imwrite(os.path.join('look','bbox_'+file_name), bbox_image)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--cate2label', default='config.yaml', help='cate2label')
    parser.add_argument('--coco_json', default='text_detect-85.json', help='input coco file')
    parser.add_argument('--output', default='gt', help='input coco file')
    main(parser.parse_args())
