#/usr/bin/python3
import os
import argparse
import git
import yaml

parser = argparse.ArgumentParser(description='''
Run evaluation of algorithm
''')
parser.add_argument('experiment_name', help='Name for the experiment')
parser.add_argument('sequence_id', help='Id of the sequence to run on)',default='rgbd_dataset_freiburg1_desk2')
parser.add_argument('--sequence_root', help='Root folder for sequences',default='/media/data/dataset/')
parser.add_argument('--run_algo', help='Set to create algorithm results',action="store_true")
args = parser.parse_args()

repo_dir = '/workspaces/ws/src/vslam_ros'
output_dir = args.sequence_root + "/" + args.sequence_id + "/" + args.experiment_name
algo_traj = output_dir + "/" + args.sequence_id+"-algo.txt"
rpe_plot = output_dir + "/" + "rpe.png"
ate_plot = output_dir + "/" + "ate.png"
xy_plot = output_dir + "/" + "xy.png"
z_plot = output_dir + "/" + "z.png"
rpe_txt = os.path.join(output_dir,'rpe.txt')
ate_txt = os.path.join(output_dir,'ate.txt')

ground_truth_traj = args.sequence_root + "/" + args.sequence_id + "/" + args.sequence_id +"-groundtruth.txt"
if not os.path.exists(output_dir):
        if not args.run_algo:
                raise ValueError("There is no algorithm output for: {}. Create it by setting --run_algo".format(args.experiment_name))
        os.makedirs(output_dir)

if args.run_algo:
        print("---------Running Algorithm-----------------")
        repo = git.Repo(repo_dir)
        sha = repo.head.object.hexsha
        with open(os.path.join(output_dir,'meta.yaml'), 'w') as f:
                yaml.dump([
                        {'name': args.experiment_name},
                        {'code_sha':sha}
                ],f)
        os.system("ros2 launch vslam_ros_evaluation evaluation.launch.py sequence_root:={} sequence_id:={} experiment_name:={}".format(args.sequence_root, args.sequence_id, args.experiment_name))
#TODO plot
print("---------Creating Plots-----------------")
os.system("python3 /workspaces/ws/src/vslam_ros/vslam_ros_evaluation/script/plot/plot_traj.py {} {} --xy_out {} --z_out {}".format(algo_traj, ground_truth_traj,xy_plot,z_plot))


print("---------Evaluating Relative Pose Error-----------------")
os.system("python3 /workspaces/ws/src/vslam_ros/vslam_ros_evaluation/script/tum/evaluate_rpe.py {} {} --verbose --plot {} --fixed_delta --delta_unit s --save {}".format(algo_traj, ground_truth_traj,rpe_plot,rpe_txt))

print("---------Evaluating Average Trajectory Error------------")
os.system("python3 /workspaces/ws/src/vslam_ros/vslam_ros_evaluation/script/tum/evaluate_ate.py {} {} --verbose --plot {} --save {}".format(ground_truth_traj, algo_traj,ate_plot,ate_txt))


# TODO upload to WandB