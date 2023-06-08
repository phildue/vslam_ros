# Copyright 2022 Philipp.Duernay
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

from ament_index_python import get_package_share_directory
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode
from launch_ros.actions import Node
import os
import shutil
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.actions import OpaqueFunction
from launch.substitutions import LaunchConfiguration

def create_algo_node(use_sim_time):
   params_algo = os.path.join(get_package_share_directory('vslam_ros'), 'config', 'NodeMapping.yaml')
   node = ComposableNode(
        package='vslam_ros',
        plugin='vslam_ros::NodeMapping',
        name='mapping',
        # remappings=[('/image', '/burgerimage')],
        parameters=[params_algo,
                    {"use_sim_time": use_sim_time},
                    {"log.config_dir": os.path.join(get_package_share_directory('vslam_ros'), 'cfg', 'log')}
                    ],
        extra_arguments=[{'use_intra_process_comms': True}])
   return node, params_algo

def create_algo_node(use_sim_time, experiment_folder):
    params_algo = os.path.join(get_package_share_directory('vslam_ros'), 'config', 'nodeMapping.yaml')
    node = ComposableNode(
        package='vslam_ros',
        plugin='vslam_ros::NodeMapping',
        name='mapping',
        # remappings=[('/image', '/burgerimage')],
        parameters=[params_algo,
                    {"use_sim_time": use_sim_time},
                    {"log.config_dir": os.path.join(get_package_share_directory('vslam_ros'), 'config', 'log')},
                    {"log.root_dir": os.path.join(experiment_folder, 'log')},
                    ],
        extra_arguments=[{'use_intra_process_comms': True}])
    return node, params_algo


def ld_opaque(context):
    use_sim_time = True
    sequence_root = LaunchConfiguration('sequence_root').perform(context)
    sequence_id = LaunchConfiguration('sequence_id').perform(context)
    experiment_name = LaunchConfiguration('experiment_name').perform(context)
    launch_without_algo = LaunchConfiguration('launch_without_algo').perform(context) == 'True'

    sequence_folder = os.path.join(sequence_root, sequence_id)
    experiment_folder = os.path.join(sequence_folder, 'algorithm_results', experiment_name)
    if not os.path.exists(experiment_folder):
        os.makedirs(experiment_folder)
    gt_traj_file = os.path.join(sequence_folder, sequence_id + "-groundtruth.txt")
    algo_traj_file = os.path.join(experiment_folder, sequence_id + "-algo.txt")
    bag_file = os.path.join(sequence_folder, sequence_id + ".db3")
    composable_nodes = [
        ComposableNode(
            package='vslam_ros',
            plugin='vslam_ros::NodeResultWriter',
            name='resultWriter',
            # remappings=[('/image', '/burgerimage')],
            parameters=[{'algoOutputFile': algo_traj_file},
                        {"use_sim_time": use_sim_time}],
            extra_arguments=[{'use_intra_process_comms': False}]),
        ComposableNode(
            package='vslam_ros',
            plugin='vslam_ros::NodeReplayer',
            name='replayer',
            # remappings=[('/image', '/burgerimage')],
            parameters=[{'bag_file': bag_file},
                        {"use_sim_time": use_sim_time},
                        {'timeout': 100},
                        {'delay': 0.1},
                        {'duration': -1.0},
                        ],
            extra_arguments=[{'use_intra_process_comms': False}]),
        ComposableNode(
            package='vslam_ros',
            plugin='vslam_ros::NodeGtLoader',
            name='nodeEvaluation',
            # remappings=[('/image', '/burgerimage')],
            parameters=[  
                  #  {"log.config_dir": os.path.join(get_package_share_directory('vslam_ros'), 'config', 'log')},
                  #  {"log.root_dir": os.path.join(experiment_folder, 'log')},
                  #  {"log.image.TrajectoryCovariance.save": False},
                  #  {"log.image.Trajectory.save": False},
                    {'gtTrajectoryFile': gt_traj_file},
                    {"use_sim_time": use_sim_time}
                        ],
            extra_arguments=[{'use_intra_process_comms': False}])
    ]
    algo_node, params_algo = create_algo_node(use_sim_time, experiment_folder)

    if not launch_without_algo:
        composable_nodes.append(algo_node)
    shutil.copy(params_algo, os.path.join(experiment_folder, 'params_algo.yaml'))

    container = ComposableNodeContainer(
        name='vslam_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container_mt',
        composable_node_descriptions=composable_nodes,
        output='both',
    )
    return [container]
    """
    node_eval = Node(
            package='vslam_ros',
            executable='nodeEvaluation',
            name='nodeEvaluation',
            parameters=[
                    {"log.config_dir": os.path.join(get_package_share_directory('vslam_ros'), 'config', 'log')},
                    {"log.root_dir": os.path.join(experiment_folder, 'log')},
                    {"log.image.TrajectoryCovariance.save": False},
                    {"log.image.Trajectory.save": False},
                    {'gtTrajectoryFile': gt_traj_file},
                    {"use_sim_time": use_sim_time}
                    ],
            remappings=[('/path', '/path/gt')])
    return [container, node_eval]
    """


def generate_launch_description():
    """Generate launch description with multiple components."""

    return LaunchDescription([
        DeclareLaunchArgument(
            'sequence_root',
            default_value=os.path.join('/mnt', 'dataset', 'tum_rgbd'),
            description='Root folder of sequences'),
        DeclareLaunchArgument(
            'experiment_name',
            default_value='experiment',
            description='Name for the experiment'),
        DeclareLaunchArgument(
            'sequence_id',
            default_value="rgbd_dataset_freiburg2_desk",
            description='id of the sequence'),
        DeclareLaunchArgument(
            'launch_without_algo',
            default_value="False",
            description='start pipeline without algorithm node'),
        OpaqueFunction(function=ld_opaque)
    ])
