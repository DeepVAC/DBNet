import sys
import numpy as np
from PIL import Image
from torch.utils import data
import cv2
import random
from shapely.geometry import Polygon
import pyclipper
import torchvision.transforms as transforms
import torch
import os

from deepvac.datasets import OsWalkDataset
from deepvac.utils import addUserConfig

random.seed(123456)

def get_img(img_path):
    try:
        img = cv2.imread(img_path)
        img = img[:, :, [2, 1, 0]]
    except Exception as e:
        print (img_path)
        raise
    return img

def get_bboxes(img, gt_path):
    h, w = img.shape[0:2]
    lines = open(gt_path).readlines()
    bboxes = []
    tags = []
    for line in lines:
        #line = util.str.remove_all(line, '\xef\xbb\xbf')
        line = line.replace('\xef\xbb\xbf', '')
        line = line.replace('\ufeff','')
        gt = line.split(',')
        #assert len(gt) %2 == 0, 'Anno points error.'

        bbox = [np.int(float(gt[i])) for i in range(len(gt)-1)]
        bbox = np.asarray(bbox) / ([w * 1.0, h * 1.0] * ((len(gt)-1)//2))

        bboxes.append(bbox)
        if len(gt[-1]) != 0 and gt[-1][0] == '#':
            tags.append(False)
        else:
            tags.append(True)
    return bboxes, tags

def random_horizontal_flip(imgs):
    if random.random() < 0.5:
        for i in range(len(imgs)):
            imgs[i] = np.flip(imgs[i], axis=1).copy()
    return imgs

def random_rotate(imgs):
    max_angle = 10
    angle = random.random() * 2 * max_angle - max_angle
    for i in range(len(imgs)):
        img = imgs[i]
        w, h = img.shape[:2]
        rotation_matrix = cv2.getRotationMatrix2D((h / 2, w / 2), angle, 1)
        img_rotation = cv2.warpAffine(img, rotation_matrix, (h, w))
        imgs[i] = img_rotation
    return imgs

def scale(img, long_size=2240):
    h, w = img.shape[0:2]
    scale = long_size * 1.0 / max(h, w)
    img = cv2.resize(img, dsize=None, fx=scale, fy=scale)
    return img

def random_scale(img, min_size):
    h, w = img.shape[0:2]
    if max(h, w) > 1280:
        scale = 1280.0 / max(h, w)
        img = cv2.resize(img, dsize=None, fx=scale, fy=scale)

    h, w = img.shape[0:2]
    random_scale = np.array([0.5, 1.0, 2.0, 3.0])
    scale = np.random.choice(random_scale)
    if min(h, w) * scale <= min_size:
        scale = (min_size + 10) * 1.0 / min(h, w)
    img = cv2.resize(img, dsize=None, fx=scale, fy=scale)
    return img

def random_crop(imgs, img_size):
    h, w = imgs[0].shape[0:2]
    th, tw = img_size
    if w == tw and h == th:
        return imgs
    
    if random.random() > 3.0 / 8.0 and np.max(imgs[1]) > 0:
        tl = np.min(np.where(imgs[1] > 0), axis = 1) - img_size
        tl[tl < 0] = 0
        br = np.max(np.where(imgs[1] > 0), axis = 1) - img_size
        br[br < 0] = 0
        br[0] = min(br[0], h - th)
        br[1] = min(br[1], w - tw)
        
        i = random.randint(tl[0], br[0])
        j = random.randint(tl[1], br[1])
    else:
        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
    
    # return i, j, th, tw
    th -= th%4
    tw -= tw%4
    for idx in range(len(imgs)):
        if len(imgs[idx].shape) == 3:
            imgs[idx] = imgs[idx][i:i + th, j:j + tw, :]
        else:
            imgs[idx] = imgs[idx][i:i + th, j:j + tw]
    return imgs

def dist(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

def perimeter(bbox):
    peri = 0.0
    for i in range(bbox.shape[0]):
        peri += dist(bbox[i], bbox[(i + 1) % bbox.shape[0]])
    return peri


def shrink_polygon_pyclipper(polygon, shrink_ratio):
    polygon_shape = Polygon(polygon)
    distance = polygon_shape.area * (1 - np.power(shrink_ratio, 2)) / polygon_shape.length
    subject = [tuple(l) for l in polygon]
    padding = pyclipper.PyclipperOffset()
    padding.AddPath(subject, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    shrinked = padding.Execute(-distance)
    if shrinked == []:
        shrinked = np.array(shrinked)
    else:
        shrinked = np.array(shrinked[0]).reshape(-1, 2)
    return shrinked



def Distance(xs, ys, point_1, point_2):
    '''
    compute the distance from point to a line
    ys: coordinates in the first axis
    xs: coordinates in the second axis
    point_1, point_2: (x, y), the end of the line
    '''
    height, width = xs.shape[:2]
    square_distance_1 = np.square(xs - point_1[0]) + np.square(ys - point_1[1])
    square_distance_2 = np.square(xs - point_2[0]) + np.square(ys - point_2[1])
    square_distance = np.square(point_1[0] - point_2[0]) + np.square(point_1[1] - point_2[1])

    cosin = (square_distance - square_distance_1 - square_distance_2) / (2 * np.sqrt(square_distance_1 * square_distance_2))
    square_sin = 1 - np.square(cosin)
    square_sin = np.nan_to_num(square_sin)

    result = np.sqrt(square_distance_1 * square_distance_2 * square_sin / square_distance)
    result[cosin < 0] = np.sqrt(np.fmin(square_distance_1, square_distance_2))[cosin < 0]
    return result


def draw_border_map(polygon, canvas, mask):
    polygon = np.array(polygon)
    assert polygon.ndim == 2
    assert polygon.shape[1] == 2

    polygon_shape = Polygon(polygon)
    if polygon_shape.area <= 0:
        return
    distance = polygon_shape.area * (1 - np.power(0.4, 2)) / polygon_shape.length
    subject = [tuple(l) for l in polygon]
    padding = pyclipper.PyclipperOffset()
    padding.AddPath(subject, pyclipper.JT_ROUND,
                    pyclipper.ET_CLOSEDPOLYGON)

    padded_polygon = np.array(padding.Execute(distance)[0])
    cv2.fillPoly(mask, [padded_polygon.astype(np.int32)], 1.0)

    xmin = padded_polygon[:, 0].min()
    xmax = padded_polygon[:, 0].max()
    ymin = padded_polygon[:, 1].min()
    ymax = padded_polygon[:, 1].max()
    width = xmax - xmin + 1
    height = ymax - ymin + 1

    polygon[:, 0] = polygon[:, 0] - xmin
    polygon[:, 1] = polygon[:, 1] - ymin

    xs = np.broadcast_to(
        np.linspace(0, width - 1, num=width).reshape(1, width), (height, width))
    ys = np.broadcast_to(
        np.linspace(0, height - 1, num=height).reshape(height, 1), (height, width))

    distance_map = np.zeros(
        (polygon.shape[0], height, width), dtype=np.float32)
    for i in range(polygon.shape[0]):
        j = (i + 1) % polygon.shape[0]
        absolute_distance = Distance(xs, ys, polygon[i], polygon[j])
        distance_map[i] = np.clip(absolute_distance / distance, 0, 1)
    distance_map = distance_map.min(axis=0)

    xmin_valid = min(max(0, xmin), canvas.shape[1] - 1)
    xmax_valid = min(max(0, xmax), canvas.shape[1] - 1)
    ymin_valid = min(max(0, ymin), canvas.shape[0] - 1)
    ymax_valid = min(max(0, ymax), canvas.shape[0] - 1)
    canvas[ymin_valid:ymax_valid + 1, xmin_valid:xmax_valid + 1] = np.fmax(
        1 - distance_map[
            ymin_valid - ymin:ymax_valid - ymax + height,
            xmin_valid - xmin:xmax_valid - xmax + width],
        canvas[ymin_valid:ymax_valid + 1, xmin_valid:xmax_valid + 1])

class DBTrainDataset(data.Dataset):
    def __init__(self, deepvac_config, data_dir, gt_dir, is_transform, img_size):
        self.config = deepvac_config.datasets
        self.transform = self.config.transform
        self.composer = self.config.composer
        
        self.is_transform = is_transform
        self.img_size = img_size if (img_size is None or isinstance(img_size, tuple)) else (img_size, img_size)
        self.shrink_ratio = addUserConfig("shrink_ratio", self.config.shrink_ratio, 0.4)
        self.thresh_min = addUserConfig("thresh_min", self.config.thresh_min, 0.3)
        self.thresh_max = addUserConfig("thresh_max", self.config.thresh_max, 0.7)
        data_dirs = [data_dir]
        gt_dirs = [gt_dir]
        self.img_paths = []
        self.gt_paths = []

        for data_dir, gt_dir in zip(data_dirs, gt_dirs):
            img_names = os.listdir(data_dir)

            img_paths = []
            gt_paths = []
            for idx, img_name in enumerate(img_names):
                img_path = data_dir + img_name
                img_paths.append(img_path)

                gt_name = 'gt_' + img_name[:-4] + '.txt'
                gt_path = gt_dir + gt_name
                gt_paths.append(gt_path)

            self.img_paths.extend(img_paths)
            self.gt_paths.extend(gt_paths)

    def __len__(self):
        return len(self.img_paths)

    def init_map_mask(self, img, tags, bboxes):
        shrink_map = np.zeros(img.shape[0:2], dtype=np.float32)
        shrink_mask = np.ones(img.shape[0:2], dtype=np.float32)
        threshold_map = np.zeros(img.shape[0:2], dtype=np.float32)
        threshold_mask = np.zeros(img.shape[0:2], dtype=np.float32)
        if len(bboxes)<=0:
            return shrink_map, shrink_mask, threshold_map, threshold_mask
            
        for i, box in enumerate(bboxes):
            bboxes[i] = np.array(box*([img.shape[1],img.shape[0]]*(len(box)//2))).reshape(len(box)//2, 2).astype('int32')
        for i, box in enumerate(bboxes):
            height = max(box[:, 1]) - min(box[:, 1])
            width = max(box[:, 0]) - min(box[:, 0])
            if not tags[i] or min(height, width) < 8:
                cv2.fillPoly(shrink_mask, [box], 0)
                continue
                
            shrinked = shrink_polygon_pyclipper(box, self.shrink_ratio)
            if shrinked.size == 0:
                cv2.fillPoly(shrink_mask, [box], 0)
                continue
            cv2.fillPoly(shrink_map, [shrinked.astype(np.int32)], 1)

        for i, box in enumerate(bboxes):
            if not tags[i]:
                continue
            draw_border_map(box, threshold_map, mask=threshold_mask)
        threshold_map = threshold_map * (self.thresh_max - self.thresh_min) + self.thresh_min

        return shrink_map, shrink_mask, threshold_map, threshold_mask

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        gt_path = self.gt_paths[index]

        img = get_img(img_path)
        bboxes, tags = get_bboxes(img, gt_path)
        
        if self.is_transform:
            img = random_scale(img, self.img_size[0])

        shrink_map, shrink_mask, threshold_map, threshold_mask = self.init_map_mask(img, tags, bboxes)
        
        if self.is_transform:
            imgs = [img, shrink_map, shrink_mask, threshold_map, threshold_mask]

            imgs = random_horizontal_flip(imgs)
            imgs = random_rotate(imgs)
            imgs = random_crop(imgs, self.img_size)

            img, shrink_map, shrink_mask, threshold_map, threshold_mask = imgs[0], imgs[1], imgs[2], imgs[3], imgs[4]
        
        if self.is_transform:
            img = Image.fromarray(img)
            img = img.convert('RGB')
            img = transforms.ColorJitter(brightness = 32.0 / 255, saturation = 0.5)(img)
        else:
            img = Image.fromarray(img)
            img = img.convert('RGB')

        img = transforms.ToTensor()(img)
        img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)

        shrink_map = torch.from_numpy(shrink_map).float()
        shrink_mask = torch.from_numpy(shrink_mask).float()
        threshold_map = torch.from_numpy(threshold_map).float()
        threshold_mask = torch.from_numpy(threshold_mask).float()
        
        return img, [shrink_map, shrink_mask, threshold_map, threshold_mask]

class DBTestDataset(OsWalkDataset):
    def __init__(self, deepvac_config, data_dir, long_size = 1280):
        super(DBTestDataset, self).__init__(deepvac_config, data_dir)
        self.long_size = long_size
    
    def scale(self, img):
        h, w = img.shape[0:2]
        scale = self.long_size * 1.0 / max(h, w)
        h, w = int(h*scale), int(w*scale)
        h += h%4
        w += w%4
        img = cv2.resize(img, (w, h))
        return img

    def __getitem__(self, idx):
        img = super(DBTestDataset, self).__getitem__(idx)
        org_img = img.copy()

        img = img[:, :, [2, 1, 0]]
        scaled_img = self.scale(img)
        scaled_img = Image.fromarray(scaled_img)
        scaled_img = scaled_img.convert('RGB')
        scaled_img = transforms.ToTensor()(scaled_img)
        scaled_img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(scaled_img)
        return org_img, scaled_img
