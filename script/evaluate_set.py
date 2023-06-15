#!/usr/bin/python3
import os
import sys
import argparse
from vslampy.evaluation.tum import TumRgbd
from evaluate import evaluate_sequence

script_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(script_dir)
parser = argparse.ArgumentParser(
    description="""
Run evaluation of algorithm"""
)
parser.add_argument("--experiment_name", help="Name for the experiment", default="fix_replay")
parser.add_argument(
    "--sequence_root", help="Root folder for sequences", default="/mnt/dataset/tum_rgbd"
)
parser.add_argument(
    "--out_root",
    help="Root folder for generating output, defaults to subfolder of sequence",
    default="",
)
parser.add_argument(
    "--upload", help="Upload results to experiment tracking tool", action="store_true"
)
parser.add_argument(
    "--override", help="Upload results to experiment tracking tool", action="store_true"
)
parser.add_argument(
    "--commit_hash", help="Id to identify algorithm version", default=""
)
parser.add_argument(
    "--workspace_dir",
    help="Directory of repository (only applicable if commit_hash not given)",
    default="/home/ros/vslam_ros",
)
args = parser.parse_args()

sequences = [
            "rgbd_dataset_freiburg1_desk",
            # "rgbd_dataset_freiburg1_desk_validation",
            "rgbd_dataset_freiburg1_desk2",
            # "rgbd_dataset_freiburg1_desk2_validation",
            "rgbd_dataset_freiburg1_floor",
            #"rgbd_dataset_freiburg1_room",
            "rgbd_dataset_freiburg1_rpy",
            #"rgbd_dataset_freiburg1_teddy",
            "rgbd_dataset_freiburg1_xyz",
            "rgbd_dataset_freiburg1_360",
            "rgbd_dataset_freiburg2_desk",
            # "rgbd_dataset_freiburg2_desk_validation",
            # "rgbd_dataset_freiburg2_pioneer_360",
            "rgbd_dataset_freiburg3_long_office_household",
            "rgbd_dataset_freiburg2_pioneer_slam",
]

failed_sequences = []
successful_sequences = []
for i, sequence in enumerate(sequences):
    try:
        evaluate_sequence(sequence_id=sequence,
                        experiment_name=args.experiment_name,
                        commit_hash=args.commit_hash,
                        workspace_dir=args.workspace_dir,
                        upload=args.upload,
                        override=True,
                        intermediate_evaluation=False
                        )
        successful_sequences += [sequence]
    except Exception as e:
        print(e)
        failed_sequences += [sequence]
    
    s = "\n - ".join(successful_sequences)
    ss = "\n - ".join(failed_sequences)
    print(f"Ran evaluation: {args.experiment_name.replace('/','_')} for sequence: {i+1}/{len(sequences)}. \nSuccessful : \n-{s}\nFailed: \n-{ss}")

