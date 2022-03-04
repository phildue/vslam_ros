#/usr/bin/python3
import os
import argparse
run_algo = False
launch_rviz = False
sequence_root = "/media/data/dataset/"
sequence_id = "rgbd_dataset_freiburg1_desk2"
output_dir = sequence_root + "/" + sequence_id
algo_traj = output_dir + "/" + "algo.txt"
rpe_plot = output_dir + "/" + "rpe.png"
ate_plot = output_dir + "/" + "ate.png"

ground_truth_traj = sequence_root + "/" + sequence_id + "/" + sequence_id +"-groundtruth.txt"

if launch_rviz:
        os.system("rviz2 &")

if run_algo:
        print("---------Running Algorithm-----------------")
        os.system("/workspaces/ws/install/bin/evaluation_app {} {} {}".format(sequence_root, sequence_id, algo_traj))
#TODO plot
print("---------Evaluating Relative Pose Error-----------------")
os.system("python3 /workspaces/ws/src/vslam_ros/vslam_ros_evaluation/script/tum/evaluate_rpe.py {} {} --verbose --plot {} --fixed_delta".format(algo_traj, ground_truth_traj,rpe_plot))

print("---------Evaluating Average Trajectory Error------------")
os.system("python3 /workspaces/ws/src/vslam_ros/vslam_ros_evaluation/script/tum/evaluate_ate.py {} {} --verbose --plot {}".format(algo_traj, ground_truth_traj,ate_plot))


