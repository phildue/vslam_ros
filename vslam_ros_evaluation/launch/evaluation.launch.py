from ament_index_python import get_package_share_directory
import launch
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode
import os
import shutil
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.actions import OpaqueFunction
from launch.substitutions import LaunchConfiguration

def ld_opaque(context):
    use_sim_time = True
    sequence_root = LaunchConfiguration('sequence_root').perform(context)
    sequence_id = LaunchConfiguration('sequence_id').perform(context)
    experiment_name = LaunchConfiguration('experiment_name').perform(context)

    sequence_folder = os.path.join(sequence_root,sequence_id)
    experiment_folder = os.path.join(sequence_folder,experiment_name)
    if not os.path.exists(experiment_folder):
        os.makedirs(experiment_folder)
    gt_traj_file = os.path.join(sequence_folder,sequence_id + "-groundtruth.txt")
    algo_traj_file = os.path.join(experiment_folder,sequence_id + "-algo.txt")
    bag_file = os.path.join(sequence_folder,sequence_id + ".db3")
    params_algo = os.path.join(get_package_share_directory('vslam_ros'),'config','RgbdAlignmentNode.yaml')
    shutil.copy(params_algo,os.path.join(experiment_folder,'params_algo.yaml'))
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
                    {"use_sim_time":use_sim_time},
                    {'timeout':10}
                    ],
                    extra_arguments=[{'use_intra_process_comms': True}]),
            ],
            output='both',
    )

    return [container]

def generate_launch_description():
    """Generate launch description with multiple components."""
    
    return LaunchDescription([
    DeclareLaunchArgument(
      'sequence_root',
      default_value=os.path.join('/media','data','dataset'),
      description='Root folder of sequences'),   
    DeclareLaunchArgument(
      'experiment_name',
      default_value='experiment',
      description='Name for the experiment'),
    DeclareLaunchArgument(
      'sequence_id',
      default_value="rgbd_dataset_freiburg1_xyz",
      description='id of the sequence'),   

    OpaqueFunction(function=ld_opaque)  
  ])

