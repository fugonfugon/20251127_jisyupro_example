#!/usr/bin/env pytohn

import rospy
from sound_play.libsoundplay import SoundClient

rospy.init_node('say_node')

rospy.sleep(1)
client = SoundClient(sound_action='robotsound_jp', sound_topic='robotsound_jp')

rospy.sleep(1)
client.say('こんにちは', voice='四国めたん-あまあま')

"""
cd ros_ws/src/
git clone --filter=blob:none --sparse https://github.com/jsk-ros-pkg/jsk_3rdparty
cd jsk_3rdparty/
git sparse-checkout set 3rdparty/voicevox/
cd  3rdparty/voicevox/
source /opt/ros/one/setup.bash 

rosdep install --from-path . --ignore-src 
catkin bt -vi
source ~/ros_ws/devel/setup.bash 
roslaunch voicevox voicevox_texttospeech.launch


"""