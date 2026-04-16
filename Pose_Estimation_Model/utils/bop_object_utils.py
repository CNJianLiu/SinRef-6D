
import os
import glob
import random
import cv2
import torchvision.transforms as transforms
import numpy as np
from tqdm import tqdm

from data_utils import (
    get_bbox,
    get_point_cloud_from_depth,
    get_resize_rgb_choose,
    load_im,
)

def generate_mask_from_alpha(image_path, mask_image_path):
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    if image.shape[2] == 4:
        alpha_channel = image[:, :, 3]
        mask = np.where(alpha_channel > 0, 255, 0).astype(np.uint8)
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(mask_image_path, mask)

class Obj:
    def __init__(self, obj_id, template_path: str, n_template_view: int):
        self.obj_id = obj_id
        self._get_template(template_path, n_template_view)

    def get_item(self, return_color=False, sample_num=2048):
        model_points = self.template_pts_r[0]
        symmetry_flag = 0
        if return_color:
            raise NotImplementedError("Colorized model export is not available in the template-only loader.")
        return model_points, symmetry_flag

    
    def _get_template(self, path, nView):
        if nView <= 0:
            self.template_choose = None
            self.template_pts_r = None
            return

        total_views = len(glob.glob(os.path.join(path, 'rgb_*.png')))
        self.template_rgb = []
        self.template_pts_r = []
        self.template_choose = []
        self.n_sample_observed_point = 2048
        self.rgb_mask_flag = True
        self.img_size = 224
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        i = random.randrange(50, 120)
        v = f"{i:03d}"
        rgb_path = os.path.join(path, '000' + str(v) + '.png')
        depth_path = os.path.join(path, '000' + str(v) + '_depth.png')
        pose_path = glob.glob(os.path.join(path, '*.npy'))[0]
        mask_path = os.path.join(path, '000' + str(v) + '_mask.png')

        generate_mask_from_alpha(rgb_path, mask_path)

        mask = load_im(mask_path).astype(np.uint8) == 255
        bbox = get_bbox(mask)
        y1, y2, x1, x2 = bbox
        H, W = mask.shape
        y1, y2 = max(0, y1), min(H, y2)
        x1, x2 = max(0, x1), min(W, x2)
        mask = mask[y1:y2, x1:x2]
        rgb = load_im(rgb_path).astype(np.uint8)[..., ::-1][y1:y2, x1:x2, :]
        rgb = rgb[:, :, :3]
        if self.rgb_mask_flag:
            rgb = rgb * (mask[:, :, None] > 0).astype(np.uint8)
        rgb = cv2.resize(rgb, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        rgb = self.transform(np.array(rgb))

        choose = mask.astype(np.float32).flatten().nonzero()[0]
        depth = load_im(depth_path).astype(np.float32) / 1000.0

        K = np.array([572.4114, 0.0, 320, 0.0, 573.57043, 240, 0.0, 0.0, 1.0]).reshape((3, 3))
        pts_r = get_point_cloud_from_depth(depth, K, [y1, y2, x1, x2])
        pts_r = pts_r.reshape(-1, 3)[choose, :]
        if len(choose) <= self.n_sample_observed_point:
            choose_idx = np.random.choice(np.arange(len(choose)), self.n_sample_observed_point)
        else:
            choose_idx = np.random.choice(np.arange(len(choose)), self.n_sample_observed_point, replace=False)
        choose = choose[choose_idx]
        pts_r = pts_r[choose_idx]
        choose = get_resize_rgb_choose(choose, [y1, y2, x1, x2], self.img_size)
        pose = np.load(pose_path).astype(np.float32)
        pose = pose[i][:4, :4]
        pose = pose.reshape(4, 4).astype(np.float32)
        rotation_matrix = pose[:3, :3]
        translation_matrix = pose[:3, 3] / 1000
        pts_r = (pts_r - translation_matrix) @ rotation_matrix

        self.template_rgb.append(rgb)
        self.template_pts_r.append(pts_r)
        self.template_choose.append(choose)

    def get_template(self):
        return self.template_rgb[0], self.template_pts_r[0], self.template_choose[0]


def load_obj(
        model_path, obj_id: int, sample_num: int,
        template_path: str,
        n_template_view: int,
):
    return Obj(obj_id, template_path, n_template_view)


def load_objs(
        model_path='models',
        template_path='templates',
        sample_num=512,
        n_template_view=0,
        show_progressbar=True
):
    objs = []
    obj_ids = sorted([int(p.split('/')[-1][4:10]) for p in glob.glob(os.path.join(model_path, '*.ply'))])

    if n_template_view>0:
        template_paths = sorted(glob.glob(os.path.join(template_path, '*')))
        assert len(template_paths) == len(obj_ids), '{} template_paths, {} obj_ids'.format(len(template_paths), len(obj_ids))
    else:
        template_paths = [None for _ in range(len(obj_ids))]

    cnt = 0
    for obj_id in tqdm(obj_ids, 'loading objects') if show_progressbar else obj_ids:
        objs.append(
            load_obj(model_path, obj_id, sample_num,
                    template_paths[cnt], n_template_view)
        )
        cnt+=1
    return objs, obj_ids
    

