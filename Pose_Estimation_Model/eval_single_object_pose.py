import argparse
import csv
import json
import os

import numpy as np
import trimesh
from tqdm import tqdm


def load_ply_model(model_path, obj_id, sample_num=2048):
    mesh = trimesh.load_mesh(os.path.join(model_path, f'obj_{obj_id:06d}.ply'))
    return mesh.sample(sample_num).astype(np.float32) / 1000.0


def transform_points(points, rotation, translation):
    return np.dot(rotation, points.T).T + translation


def compute_add_error(estimated_r, estimated_t, gt_r, gt_t, points):
    estimated_points = transform_points(points, estimated_r, estimated_t)
    gt_points = transform_points(points, gt_r, gt_t)
    return np.linalg.norm(estimated_points - gt_points, axis=1).mean()


def compute_adds_error(estimated_r, estimated_t, gt_r, gt_t, points):
    estimated_points = transform_points(points, estimated_r, estimated_t)
    gt_points = transform_points(points, gt_r, gt_t)
    add_s_errors = []
    for point in estimated_points:
        add_s_errors.append(np.min(np.linalg.norm(gt_points - point, axis=1)))
    return np.mean(add_s_errors)


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
                    ),
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
                    't': np.array(obj['cam_t_m2c']),
                }
            )
    return gt_poses


def load_model_info(model_info_file):
    with open(model_info_file, 'r') as file:
        return json.load(file)


def cal_auc_cdf(errors, max_threshold=0.1, step=0.005):
    thresholds = np.arange(0, max_threshold + step, step)
    success_rates = [np.mean(errors <= threshold) for threshold in thresholds]
    return np.mean(success_rates) * 100


def get_parser():
    parser = argparse.ArgumentParser(description="Evaluate per-object ADD and ADD-S AUC for a selected object.")
    parser.add_argument("--result_csv", required=True, help="Path to predicted pose csv.")
    parser.add_argument("--models_dir", required=True, help="Path to BOP models directory.")
    parser.add_argument("--test_dir", required=True, help="Path to BOP test split directory.")
    parser.add_argument("--models_info", required=True, help="Path to models_info.json.")
    parser.add_argument("--obj_id", type=int, required=True, help="Target object id.")
    return parser


def main():
    args = get_parser().parse_args()

    estimated_poses = load_estimated_poses_from_csv(args.result_csv)
    diameter_data = load_model_info(args.models_info)
    add_dis_dict = {}
    add_s_dis_dict = {}

    for pose in tqdm(estimated_poses, desc="Processing poses"):
        if pose['obj_id'] != args.obj_id:
            continue

        points = load_ply_model(args.models_dir, pose['obj_id'])
        diameter = diameter_data.get(str(pose['obj_id']), {}).get('diameter')
        if diameter is None:
            print(f"Missing diameter info for object {pose['obj_id']}.")
            continue

        gt_path = os.path.join(args.test_dir, f"{pose['scene_id']:06d}", 'scene_gt.json')
        gt_poses = load_gt_from_json(gt_path)
        gt_pose = next((obj for obj in gt_poses[str(pose['im_id'])] if obj["obj_id"] == pose['obj_id']), None)
        if gt_pose is None:
            continue

        add_error = compute_add_error(pose['R'], pose['t'], gt_pose['R'], gt_pose['t'], points) / diameter
        add_s_error = compute_adds_error(pose['R'], pose['t'], gt_pose['R'], gt_pose['t'], points) / diameter

        add_dis_dict.setdefault(pose['obj_id'], []).append(add_error)
        add_s_dis_dict.setdefault(pose['obj_id'], []).append(add_s_error)

    for obj_id in sorted(add_dis_dict.keys()):
        add_auc = cal_auc_cdf(add_dis_dict[obj_id])
        add_s_auc = cal_auc_cdf(add_s_dis_dict[obj_id])
        print(f"Object {obj_id} - ADD AUC: {add_auc:.2f}, ADD-S AUC: {add_s_auc:.2f}")


if __name__ == "__main__":
    main()
