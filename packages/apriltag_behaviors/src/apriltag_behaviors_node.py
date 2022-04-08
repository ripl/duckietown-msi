#!/usr/bin/env python3

import json
from typing import Dict

import rospy

from duckietown.dtros import DTROS, NodeType

from duckietown_msgs.msg import AprilTagDetectionArray
from duckietown_msgs.msg import BoolStamped


TagID = int


class AprilTagBehavior(DTROS):
    def __init__(self):
        super(AprilTagBehavior, self).__init__(
            node_name="apriltag_behaviors_node", node_type=NodeType.PERCEPTION
        )
        # get static parameters
        behaviors = rospy.get_param("~behaviors", [])
        self._behaviors: Dict[TagID, dict] = {b["tag"]: b for b in behaviors}
        # create subscribers
        self._sub = rospy.Subscriber(
            "~detections", AprilTagDetectionArray, self._cb, queue_size=1
        )
        # create subscribers
        self.pub_joy_override = rospy.Publisher(
            "~joystick_override", BoolStamped, queue_size=1
        )
        # internal state
        self._last_behavior = None
        # print out behaviors
        self.loginfo(f"Behaviors are: {json.dumps(self._behaviors, indent=4, sort_keys=True)}")

    def _cb(self, msg):
        for detection in msg.detections:
            new_behavior = self._behaviors.get(detection.tag_id, None)
            if new_behavior is not None and new_behavior != self._last_behavior:
                self.loginfo(f"Switched to new behavior: {new_behavior['name']}")
                self._last_behavior = new_behavior
                # publish message
                # Back button: Stop LF
                override_msg = BoolStamped()
                override_msg.header.stamp = msg.header.stamp
                override_msg.data = new_behavior['name'] != "LANE_FOLLOWING_MODE"
                self.pub_joy_override.publish(override_msg)


if __name__ == "__main__":
    node = AprilTagBehavior()
    # spin forever
    rospy.spin()
