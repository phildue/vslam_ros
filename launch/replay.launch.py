
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription

import launch_ros.actions
import launch_ros.descriptions
import launch.substitutions
#TODO put as launch params
FPS=1
CALIBRATION_PATH='/home/pi/stereo_ws/config/calibration'
WIDTH=640
HEIGHT=480
def generate_launch_description():
    return LaunchDescription([
        launch_ros.actions.Node(
            package='vslam_ros', executable='lukas_kanade_se3_node', output='screen',
            name='lukas_kanade_se3_node')
    ])

