import argparse
import csv
import json
import os

import numpy as np
import trimesh
from scipy import spatial
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
    nn_index = spatial.cKDTree(estimated_points)
    nn_dists, _ = nn_index.query(gt_points, k=1)
    return nn_dists.mean()


def add_metric(estimated_r, estimated_t, gt_r, gt_t, points, diameter):
    add_error = compute_add_error(estimated_r, estimated_t, gt_r, gt_t, points) / diameter
    return add_error < 0.1


def load_estimated_poses_from_csv(csv_file):
    estimated_poses = []
    with open(csv_file, 'r') as file:
        reader = csv.reader(file, delimiter=',')
        for row in reader:
            scene_id = int(row[0])
            im_id = int(row[1])
            obj_id = int(row[2])
            score = float(row[3])

            r_elements = row[4].split()
            rotation = np.array(
                [
                    [float(r_elements[0]), float(r_elements[1]), float(r_elements[2])],
                    [float(r_elements[3]), float(r_elements[4]), float(r_elements[5])],
                    [float(r_elements[6]), float(r_elements[7]), float(r_elements[8])],
                ],
                dtype=np.float32,
            )

            t_elements = row[5].split()
            translation = np.array(
                [float(t_elements[0]), float(t_elements[1]), float(t_elements[2])],
                dtype=np.float32,
            ) / 1000.0

            estimated_poses.append(
                {
                    'scene_id': scene_id,
                    'im_id': im_id,
                    'obj_id': obj_id,
                    'score': score,
                    'R': rotation,
                    't': translation,
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


def cal_auc(add_dis, max_dis=0.1):
    distances = np.array(add_dis)
    distances[np.where(distances > max_dis)] = np.inf
    distances = np.sort(distances)
    n = len(add_dis)
    acc = np.cumsum(np.ones((1, n)), dtype=np.float32) / n
    return voc_ap(distances, acc) * 100


def voc_ap(rec, prec):
    idx = np.where(rec != np.inf)
    if len(idx[0]) == 0:
        return 0
    rec = rec[idx]
    prec = prec[idx]
    mrec = np.array([0.0] + list(rec) + [0.1])
    mpre = np.array([0.0] + list(prec) + [prec[-1]])
    for i in range(1, prec.shape[0]):
        mpre[i] = max(mpre[i], mpre[i - 1])
    i = np.where(mrec[1:] != mrec[:-1])[0] + 1
    return np.sum((mrec[i] - mrec[i - 1]) * mpre[i]) * 10


def get_parser():
    parser = argparse.ArgumentParser(description="Evaluate LM-style results with ADD and ADD-S metrics.")
    parser.add_argument("--result_csv", required=True, help="Path to predicted pose csv.")
    parser.add_argument("--models_dir", required=True, help="Path to BOP models directory.")
    parser.add_argument("--test_dir", required=True, help="Path to BOP test split directory.")
    parser.add_argument("--models_info", required=True, help="Path to models_info.json.")
    return parser


def main():
    args = get_parser().parse_args()

    estimated_poses = load_estimated_poses_from_csv(args.result_csv)
    diameter_data = load_model_info(args.models_info)

    success_count = {}
    total_count = {}
    add_distances_count = {}
    add_s_distances_count = {}

    for pose in tqdm(estimated_poses, desc="Processing poses"):
        scene_id = pose['scene_id']
        im_id = pose['im_id']
        obj_id = pose['obj_id']
        points = load_ply_model(args.models_dir, obj_id)

        gt_path = os.path.join(args.test_dir, f'{scene_id:06d}', 'scene_gt.json')
        gt_poses = load_gt_from_json(gt_path)
        gt_pose = next((obj for obj in gt_poses[str(im_id)] if obj["obj_id"] == obj_id), None)
        if gt_pose is None:
            continue

        estimated_r = pose['R']
        estimated_t = pose['t']
        if np.allclose(estimated_r, np.eye(3)) and np.allclose(estimated_t, [0.0, 0.0, 0.0]):
            continue

        diameter = diameter_data[str(obj_id)]['diameter'] / 1000.0
        add_error = compute_add_error(estimated_r, estimated_t, gt_pose['R'], gt_pose['t'], points)
        add_s_error = compute_adds_error(estimated_r, estimated_t, gt_pose['R'], gt_pose['t'], points)

        add_distances_count.setdefault(obj_id, []).append(add_error)
        add_s_distances_count.setdefault(obj_id, []).append(add_s_error)
        success_count.setdefault(obj_id, 0)
        total_count.setdefault(obj_id, 0)

        if add_metric(estimated_r, estimated_t, gt_pose['R'], gt_pose['t'], points, diameter):
            success_count[obj_id] += 1
        total_count[obj_id] += 1

    print("\nCorrectness Percentage for each object:")
    for obj_id in sorted(success_count.keys()):
        percentage = (success_count[obj_id] / total_count[obj_id]) * 100
        print(f"Object {obj_id}: {percentage:.2f}%")

    print("\nAUC of ADD distances for each object:")
    for obj_id, distances in sorted(add_distances_count.items()):
        print(f"Object {obj_id} AUC: {cal_auc(distances):.2f}%")

    print("\nAUC of ADD-S distances for each object:")
    for obj_id, distances in sorted(add_s_distances_count.items()):
        print(f"Object {obj_id} AUC-S: {cal_auc(distances):.2f}%")


if __name__ == "__main__":
    main()
