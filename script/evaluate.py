#!/usr/bin/python3
import os
import argparse
import git
import yaml

parser = argparse.ArgumentParser(description='''
Run evaluation of algorithm''')
parser.add_argument('experiment_name', help='Name for the experiment')
parser.add_argument('sequence_id', help='Id of the sequence to run on)',
                    default='rgbd_dataset_freiburg1_desk2')
parser.add_argument('--sequence_root',
                    help='Root folder for sequences',
                    default='/mnt/dataset/tum_rgbd')
parser.add_argument('--run_algo', help='Set to create algorithm results', action="store_true")
args = parser.parse_args()

repo_dir = '/workspaces/vslam'
output_dir = args.sequence_root + "/" + args.sequence_id + "/" + args.experiment_name
algo_traj = output_dir + "/" + args.sequence_id+"-algo.txt"
rpe_plot = output_dir + "/" + "rpe.png"
ate_plot = output_dir + "/" + "ate.png"
xy_plot = output_dir + "/" + "xy.png"
z_plot = output_dir + "/" + "z.png"
rpe_txt = os.path.join(output_dir, 'rpe.txt')
ate_txt = os.path.join(output_dir, 'ate.txt')

gt_traj = os.path.join(args.sequence_root, args.sequence_id, args.sequence_id + "-groundtruth.txt")
if not os.path.exists(output_dir):
    if not args.run_algo:
        raise ValueError("There is no algorithm output for: {args.experiment_name}. \
            Create it by setting --run_algo")
    os.makedirs(output_dir)

if args.run_algo:
    print("---------Running Algorithm-----------------")
    repo = git.Repo(repo_dir)
    sha = repo.head.object.hexsha
    with open(os.path.join(output_dir, 'meta.yaml'), 'w') as f:
        yaml.dump([
                {'name': args.experiment_name},
                {'code_sha': sha}
                ], f)
    os.system(f"ros2 launch vslam_ros evaluation.launch.py \
        sequence_root:={args.sequence_root} sequence_id:={args.sequence_id} \
        experiment_name:={args.experiment_name}")
# TODO plot
print("---------Creating Plots-----------------")
os.system(f"python3 /workspaces/vslam/script/plot/plot_traj.py \
    {algo_traj} {gt_traj} --xy_out {xy_plot} --z_out {z_plot}")


print("---------Evaluating Relative Pose Error-----------------")
os.system(f"python3 /workspaces/vslam/script/tum/evaluate_rpe.py \
    {gt_traj} {algo_traj} \
    --verbose --plot {rpe_plot} --fixed_delta --delta_unit s --save {rpe_plot}")

print("---------Evaluating Average Trajectory Error------------")
os.system(f"python3 /workspaces/vslam/script/tum/evaluate_ate.py \
    {gt_traj} {algo_traj} \
    --verbose --plot {ate_plot} --save {ate_txt}")


# TODO upload to WandB
