#/usr/bin/python3
import os
import argparse
parser = argparse.ArgumentParser(description='''
Run evaluation of algorithm
''')
parser.add_argument('experiment_name', help='Name for the experiment')
parser.add_argument('sequence_id', help='Id of the sequence to run on)',default='rgbd_dataset_freiburg1_desk2')
parser.add_argument('--sequence_root', help='Root folder for sequences',default='/media/data/dataset/')
parser.add_argument('--run_algo', help='time offset added to the timestamps of the second file (default: 0.0)',action="store_true")
args = parser.parse_args()


output_dir = args.sequence_root + "/" + args.sequence_id + "/" + args.experiment_name
algo_traj = output_dir + "/" + "algo.txt"
rpe_plot = output_dir + "/" + "rpe.png"
ate_plot = output_dir + "/" + "ate.png"
xy_plot = output_dir + "/" + "xy.png"
z_plot = output_dir + "/" + "z.png"

ground_truth_traj = args.sequence_root + "/" + args.sequence_id + "/" + args.sequence_id +"-groundtruth.txt"
if not os.path.exists(output_dir):
        os.makedirs(output_dir)

if args.run_algo:
        print("---------Running Algorithm-----------------")
        os.system("/workspaces/ws/install/bin/evaluation_app {} {} {}".format(args.sequence_root, args.sequence_id, algo_traj))

#TODO plot
print("---------Creating Plots-----------------")
os.system("python3 /workspaces/ws/src/vslam_ros/vslam_ros_evaluation/script/plot/plot_traj.py {} {} --xy_out {} --z_out {}".format(algo_traj, ground_truth_traj,xy_plot,z_plot))


print("---------Evaluating Relative Pose Error-----------------")
os.system("python3 /workspaces/ws/src/vslam_ros/vslam_ros_evaluation/script/tum/evaluate_rpe.py {} {} --verbose --plot {} --fixed_delta".format(algo_traj, ground_truth_traj,rpe_plot))

print("---------Evaluating Average Trajectory Error------------")
os.system("python3 /workspaces/ws/src/vslam_ros/vslam_ros_evaluation/script/tum/evaluate_ate.py {} {} --verbose --plot {}".format(algo_traj, ground_truth_traj,ate_plot))


# TODO upload to WandB