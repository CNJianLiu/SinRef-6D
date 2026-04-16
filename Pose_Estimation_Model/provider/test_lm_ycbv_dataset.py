import os
import json
import numpy as np

import torch
import torchvision.transforms as transforms

from data_utils import (
    get_bop_depth_map,
    get_bop_image,
    get_bbox,
    get_point_cloud_from_depth,
    get_resize_rgb_choose,
    load_im,
    resolve_data_path,
)
from bop_object_utils import load_objs



class LINEMODTestset():
    def __init__(self, cfg, eval_dataset_name='ycbv'):
        self.cfg = cfg
        self.dataset = eval_dataset_name
        self.data_dir = resolve_data_path(cfg.data_dir)
        self.rgb_mask_flag = cfg.rgb_mask_flag
        self.img_size = cfg.img_size
        self.n_sample_observed_point = cfg.n_sample_observed_point
        self.n_sample_model_point = cfg.n_sample_model_point
        self.n_sample_template_point = cfg.n_sample_template_point
        self.n_template_view = cfg.n_template_view
        self.minimum_n_point = cfg.minimum_n_point
        self.seg_filter_score = cfg.seg_filter_score
        self.transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                std=[0.229, 0.224, 0.225])])

        if eval_dataset_name == 'tless':
            model_path = 'models_cad'
        else:
            model_path = 'models'
        if eval_dataset_name == 'lm_test_all':
            self.template_folder = os.path.join(resolve_data_path(cfg.template_dir), 'lm_test_all')
        if eval_dataset_name == 'ycbv_test_all':
            self.template_folder = os.path.join(resolve_data_path(cfg.template_dir), 'ycbv')
        
        self.data_folder = os.path.join(self.data_dir, eval_dataset_name, 'test')
        self.model_folder = os.path.join(self.data_dir, eval_dataset_name, model_path)
        obj, obj_ids = load_objs(self.model_folder, self.template_folder, sample_num=self.n_sample_model_point, n_template_view=self.n_template_view)
        obj_idxs = {obj_id: idx for idx, obj_id in enumerate(obj_ids)}
        self.objects = obj
        self.obj_idxs = obj_idxs

        self.det_keys = []
        self.dets = {}
        if eval_dataset_name == 'lm_test_all':
            for scene_id in range(1, 16):
                scene_id_str = f'{scene_id:06d}'
                scene_folder = os.path.join(self.data_folder, scene_id_str)
                gt_file = os.path.join(scene_folder, 'scene_gt.json')
                with open(gt_file) as f:
                    gt_data = json.load(f)

                for img_id in gt_data.keys():
                    key = scene_id_str + '_' + str(img_id).zfill(6)
                    self.det_keys.append(key)
                    self.dets[key] = gt_data[img_id]
        
        if eval_dataset_name == 'ycbv_test_all':
            for scene_id in range(48, 60):
                scene_id_str = f'{scene_id:06d}'
                scene_folder = os.path.join(self.data_folder, scene_id_str)
                gt_file = os.path.join(scene_folder, 'scene_gt.json')
                with open(gt_file) as f:
                    gt_data = json.load(f)

                for img_id in gt_data.keys():
                    key = scene_id_str + '_' + str(img_id).zfill(6)
                    self.det_keys.append(key)
                    self.dets[key] = gt_data[img_id]
        print('testing on {} images on {}...'.format(len(self.det_keys), eval_dataset_name))


    def __len__(self):
        return len(self.det_keys)
    
    def __getitem__(self, index):
        dets = self.dets[self.det_keys[index]]

        scene_id = int(self.det_keys[index][0:6])
        img_id = int(self.det_keys[index][7:13])

        instances = []
        for det in dets:
            instance = self.get_instance(det, scene_id, img_id)
            if instance is not None:
                instances.append(instance)

        if len(instances) == 0:
            print(f"No valid instances found for index {index}. Skipping this data point.")
            return {}

        ret_dict = {}
        for key in instances[0].keys():
            ret_dict[key] = torch.stack([instance[key] for instance in instances])
        ret_dict['scene_id'] = torch.IntTensor([scene_id])
        ret_dict['img_id'] = torch.IntTensor([img_id])

        return ret_dict

    def get_instance(self, data, scene_id, img_id):
        obj_id = data['obj_id']
        scene_folder = os.path.join(self.data_folder, f'{scene_id:06d}')

        scene_camera = json.load(open(os.path.join(scene_folder, 'scene_camera.json')))
        K = np.array(scene_camera[str(img_id)]['cam_K']).reshape((3, 3)).copy()
        depth_scale = scene_camera[str(img_id)]['depth_scale']
        inst = dict(scene_id=scene_id, img_id=img_id, data_folder=self.data_folder)
        depth = get_bop_depth_map(inst) 
        depth = depth * depth_scale
        mask_path = os.path.join(scene_folder, 'mask', f'{img_id:06d}_000000.png')
        mask = load_im(mask_path).astype(np.uint8) == 255
        if mask is None or np.sum(mask) <= self.minimum_n_point:
            return None

        bbox = get_bbox(mask)
        y1, y2, x1, x2 = bbox
        mask = mask[y1:y2, x1:x2]
        choose = mask.astype(np.float32).flatten().nonzero()[0]
        
        all_tem_rgb = {}
        all_tem_choose = {}
        all_tem_pts = {}

        for obj in self.objects:
            id = obj.obj_id
            tem_rgb, tem_pts, tem_choose = self._get_template(obj)
            all_tem_rgb[id] = tem_rgb
            all_tem_choose[id] = tem_choose
            all_tem_pts[id] = tem_pts
           
        cloud = get_point_cloud_from_depth(depth, K, [y1, y2, x1, x2])
        cloud = cloud.reshape(-1, 3)[choose, :]
        center = np.mean(cloud, axis=0)
        tmp_cloud = cloud - center[None, :]
        max_radius = np.max(np.linalg.norm(all_tem_pts[obj_id], axis=1))
        flag = np.linalg.norm(tmp_cloud, axis=1) < max_radius * 1.2
        if np.sum(flag) < self.minimum_n_point:
            return None
        choose = choose[flag]
        cloud = cloud[flag]

        if len(choose) <= self.n_sample_observed_point:
            choose_idx = np.random.choice(np.arange(len(choose)), self.n_sample_observed_point)
        else:
            choose_idx = np.random.choice(np.arange(len(choose)), self.n_sample_observed_point, replace=False)
        choose = choose[choose_idx]
        cloud = cloud[choose_idx]

        rgb,_ = get_bop_image(inst, [y1,y2,x1,x2], self.img_size, mask if self.rgb_mask_flag else None)
        ret_dict = {}
        rgb = self.transform(np.array(rgb))
        rgb_choose = get_resize_rgb_choose(choose, [y1, y2, x1, x2], self.img_size)

        ret_dict['pts'] = torch.FloatTensor(cloud)
        ret_dict['rgb'] = torch.FloatTensor(rgb)
        ret_dict['rgb_choose'] = torch.IntTensor(rgb_choose).long()
        ret_dict['obj'] = torch.IntTensor([self.obj_idxs[obj_id]]).long()
        ret_dict['obj_id'] = torch.IntTensor([obj_id])
        return ret_dict

    def _get_template(self, obj):
        rgb, pts_r, choose = obj.get_template()
        return rgb, pts_r, choose

    def get_templates(self):
        all_tem_rgb = [[]]
        all_tem_choose = [[]]
        all_tem_pts = [[]]

        for obj in self.objects:
            tem_rgb, tem_pts, tem_choose = self._get_template(obj)
            all_tem_rgb[0].append(torch.FloatTensor(tem_rgb))
            all_tem_choose[0].append(torch.IntTensor(tem_choose).long())
            all_tem_pts[0].append(torch.FloatTensor(tem_pts))

        one_rgb = [torch.stack(all_tem_rgb[0]).cuda()]
        one_pts = [torch.stack(all_tem_pts[0]).cuda()]
        one_choose = [torch.stack(all_tem_choose[0]).cuda()]

        return one_rgb, one_pts, one_choose
