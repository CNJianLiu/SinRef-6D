import argparse
import csv
import json
import os

import numpy as np
from scipy.spatial import cKDTree
from tqdm import tqdm


def load_xyz_model(model_root, obj_id):
    model_path = os.path.join(model_root, f"{obj_id}.xyz")
    return np.loadtxt(model_path, dtype=np.float32)


def transform_points(points, rotation, translation):
    return np.dot(rotation, points.T).T + translation


def add_error(estimated_r, estimated_t, gt_r, gt_t, points):
    estimated_points = transform_points(points, estimated_r, estimated_t)
    gt_points = transform_points(points, gt_r, gt_t)
    return np.linalg.norm(estimated_points - gt_points, axis=-1).mean()


def adds_error(estimated_r, estimated_t, gt_r, gt_t, points):
    estimated_points = transform_points(points, estimated_r, estimated_t)
    gt_points = transform_points(points, gt_r, gt_t)
    nn_index = cKDTree(estimated_points)
    nn_dists, _ = nn_index.query(gt_points, k=1, workers=-1)
    return nn_dists.mean()


def load_estimated_poses_from_csv(csv_file):
    estimated_poses = []
    with open(csv_file, 'r') as file:
        reader = csv.reader(file, delimiter=',')
        for row in reader:
            r_elements = row[4].split()
            t_elements = row[5].split()
            estimated_poses.append(
                {
                    'scene_id': int(row[0]),
                    'im_id': int(row[1]),
                    'obj_id': int(row[2]),
                    'score': float(row[3]),
                    'R': np.array(
                        [
                            [float(r_elements[0]), float(r_elements[1]), float(r_elements[2])],
                            [float(r_elements[3]), float(r_elements[4]), float(r_elements[5])],
                            [float(r_elements[6]), float(r_elements[7]), float(r_elements[8])],
                        ],
                        dtype=np.float32,
                    ),
                    't': np.array(
                        [float(t_elements[0]), float(t_elements[1]), float(t_elements[2])],
                        dtype=np.float32,
                    ) / 1000.0,
                }
            )
    return estimated_poses


def load_gt_from_json(gt_file):
    with open(gt_file, 'r') as file:
        gt_data = json.load(file)

    gt_poses = {}
    for image_id, objects in gt_data.items():
        gt_poses[image_id] = []
        for obj in objects:
            obj_id = obj.get('obj_id') or obj.get('obj_1d')
            if obj_id is None:
                raise ValueError(f"Failed to read obj_id from {obj}")
            gt_poses[image_id].append(
                {
                    'obj_id': obj_id,
                    'R': np.array(obj['cam_R_m2c']).reshape(3, 3),
                    't': np.array(obj['cam_t_m2c']) / 1000.0,
                }
            )
    return gt_poses


def load_model_info(model_info_file):
    with open(model_info_file, 'r') as file:
        return json.load(file)


def compute_auc(errors, max_val=0.1, step=0.001):
    errors = np.sort(np.array(errors))
    thresholds = np.arange(0, max_val + step, step)
    success = np.array([(errors <= thr).sum() / len(errors) for thr in thresholds], dtype=np.float32)
    return np.trapz(success, thresholds) / max_val * 100


def get_parser():
    parser = argparse.ArgumentParser(description="Evaluate YCB-V style results with ADD and ADD-S AUC.")
    parser.add_argument("--result_csv", required=True, help="Path to predicted pose csv.")
    parser.add_argument("--models_dir", required=True, help="Path to model point cloud directory.")
    parser.add_argument("--test_dir", required=True, help="Path to BOP test split directory.")
    parser.add_argument("--models_info", required=True, help="Path to models_info.json.")
    return parser


def main():
    args = get_parser().parse_args()

    estimated_poses = load_estimated_poses_from_csv(args.result_csv)
    diameter_data = load_model_info(args.models_info)

    add_dis_dict = {}
    add_s_dis_dict = {}

    for pose in tqdm(estimated_poses, desc="Processing poses"):
        scene_id = pose['scene_id']
        im_id = pose['im_id']
        obj_id = pose['obj_id']
        points = load_xyz_model(args.models_dir, obj_id)

        gt_path = os.path.join(args.test_dir, f'{scene_id:06d}', 'scene_gt.json')
        gt_poses = load_gt_from_json(gt_path)
        gt_pose = next((obj for obj in gt_poses[str(im_id)] if obj["obj_id"] == obj_id), None)
        if gt_pose is None:
            continue

        estimated_r = pose['R']
        estimated_t = pose['t']
        if np.allclose(estimated_r, np.eye(3)) and np.allclose(estimated_t, [0.0, 0.0, 0.0]):
            continue

        if str(obj_id) not in diameter_data:
            continue

        add_dis_dict.setdefault(obj_id, []).append(add_error(estimated_r, estimated_t, gt_pose['R'], gt_pose['t'], points))
        add_s_dis_dict.setdefault(obj_id, []).append(adds_error(estimated_r, estimated_t, gt_pose['R'], gt_pose['t'], points))

    for obj_id in sorted(add_dis_dict.keys()):
        add_auc = compute_auc(add_dis_dict[obj_id])
        add_s_auc = compute_auc(add_s_dis_dict[obj_id])
        print(f"Object {obj_id} - ADD AUC: {add_auc:.2f}, ADD-S AUC: {add_s_auc:.2f}")


if __name__ == "__main__":
    main()
