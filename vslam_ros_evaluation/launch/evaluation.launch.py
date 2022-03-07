from ament_index_python import get_package_share_directory
import launch
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode
import os

def generate_launch_description():
    """Generate launch description with multiple components."""
    use_sim_time = True
    
    #TODO make this launch parameters
    sequence_root = os.path.join('/media','data','dataset')
    experiment_name = "experiment"
    sequence_id = "rgbd_dataset_freiburg1_xyz"
    sequence_folder = os.path.join(sequence_root,sequence_id)


    if not os.path.exists(os.path.join(sequence_folder,experiment_name)):
        os.makedirs(os.path.join(sequence_folder,experiment_name))
    gt_traj_file = os.path.join(sequence_folder,sequence_id + "-groundtruth.txt")
    algo_traj_file = os.path.join(sequence_folder,experiment_name,sequence_id + "-algo.txt")
    bag_file = os.path.join(sequence_folder,sequence_id + ".db3")
    params_algo = os.path.join(get_package_share_directory('vslam_ros'),'config','RgbdAlignmentNode.yaml')
    container = ComposableNodeContainer(
            name='vslam_container',
            namespace='',
            package='rclcpp_components',
            executable='component_container',
            composable_node_descriptions=[
                ComposableNode(
                    package='vslam_ros',
                    plugin='vslam_ros::RgbdAlignmentNode',
                    name='rgbdAlignment',
                    #remappings=[('/image', '/burgerimage')],
                    parameters=[params_algo,
                    {"use_sim_time":use_sim_time}],
                    extra_arguments=[{'use_intra_process_comms': True}]),
                ComposableNode(
                    package='vslam_ros_evaluation',
                    plugin='NodeGtLoader',
                    name='gtLoader',
                    remappings=[('/path', '/path/gt')],
                    parameters=[{'gtTrajectoryFile': gt_traj_file},
                    {"use_sim_time":use_sim_time}],
                    extra_arguments=[{'use_intra_process_comms': True}]),
                ComposableNode(
                    package='vslam_ros_evaluation',
                    plugin='NodeResultWriter',
                    name='resultWriter',
                    #remappings=[('/image', '/burgerimage')],
                    parameters=[{'algoOutputFile': algo_traj_file},
                    {"use_sim_time":use_sim_time}],
                    extra_arguments=[{'use_intra_process_comms': True}]),
                ComposableNode(
                    package='vslam_ros_evaluation',
                    plugin='NodeReplayer',
                    name='replayer',
                    #remappings=[('/image', '/burgerimage')],
                    parameters=[{'bag_file': bag_file},
                    {'period': 0.2},
                    {"use_sim_time":use_sim_time}],
                    extra_arguments=[{'use_intra_process_comms': True}]),
            ],
            output='both',
    )

    return launch.LaunchDescription([container])