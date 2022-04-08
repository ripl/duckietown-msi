#!/usr/bin/env python3

import time
from collections import defaultdict
from queue import Queue, Empty
from threading import Thread
from typing import Any, Optional

# camera
import numpy as np
import rospy
from duckietown.dtros import DTROS, NodeType
from duckietown_msgs.msg import WheelsCmdStamped
from sensor_msgs.msg import CompressedImage, CameraInfo

# TODO: this should come from ROS as well, this is myrobot's
H_static = [[-2.42749970e-02, 9.46389079e-02, 3.81909422e-01],
            [-4.55028567e-01, -1.17673909e-03, -1.87813039e-02],
            [-1.46006785e-01, 3.29784838e+00, 1]]

# distance between wheels
wheel_baseline: float = 0.1
# IK parameters
v_max: float = 1.0
omega_max: float = 10.0
# calibration from TTIC's myrobot
wheel_radius = 0.0318

# IK parameters
gain: float = 1.3
k: float = 27.0
limit: float = 1.0
trim = 0.1

# x, y, w, h
CROP = [0, 200, 640, 280]

# line detector
from typing import Dict
from turbojpeg import TurboJPEG
from dt_computer_vision.line_detection import LineDetector, ColorRange, Detections

# ground projector
from typing import List
from dt_computer_vision.camera import CameraModel, NormalizedImagePoint, Pixel
from dt_computer_vision.ground_projection import GroundProjector
from dt_computer_vision.ground_projection.types import GroundPoint

# lane filter
from dt_state_estimation.lane_filter import LaneFilterHistogram
from dt_state_estimation.lane_filter.types import Segment, SegmentPoint, SegmentColor

# lane controller
from dt_motion_planning.lane_controller import PIDLaneController

# inverse kinematics
from dt_modeling.kinematics.inverse import InverseKinematics

TagID = int

jpeg = TurboJPEG()


class LaneFollowing(DTROS):
    def __init__(self):
        super(LaneFollowing, self).__init__(node_name="lane_following_node",
                                            node_type=NodeType.CONTROL)
        # create subscribers
        self._img_sub = rospy.Subscriber(
            "~image", CompressedImage, self._img_cb, queue_size=1, buff_size="20M"
        )
        self._cinfo_sub = rospy.Subscriber(
            "~camera_info", CameraInfo, self._cinfo_cb, queue_size=1
        )
        # create publishers
        self._wheels_pub = rospy.Publisher(
            "~wheels_cmd", WheelsCmdStamped, queue_size=1
        )
        # internal state
        self._camera_info = None
        # thread pool
        self._queues = defaultdict(lambda: Queue(1))
        # make worker threads
        self._workers = [
            Thread(target=self._line_detector, daemon=True),
            Thread(target=self._lane_filter, daemon=True),
            # Thread(target=self._lane_filter, daemon=True),
            Thread(target=self._lane_controller, daemon=True),
            Thread(target=self._inverse_kinematics, daemon=True),
            Thread(target=self._wheels, daemon=True),
        ]
        for w in self._workers:
            w.start()

    def _queue_pop(self, queue: str) -> Any:
        value = self._queues[queue].get()
        # self.loginfo(f"POP: {queue}")
        return value

    def _queue_full(self, queue: str):
        return self._queues[queue].full()

    def _queue_put(self, queue: str, value: Any):
        try:
            self._queues[queue].get(block=False)
        except Empty:
            pass
        self._queues[queue].put(value)
        # self.loginfo(f"PUT: {queue}")

    def _img_cb(self, msg):
        if self._queue_full("images"):
            return
        x, y, w, h = CROP
        bgr = jpeg.decode(msg.data)
        bgr = bgr[y:y + h, x:x + w, :]
        self._queue_put("image", bgr)

    def _cinfo_cb(self, msg):
        if self._camera_info is not None:
            return
        x, y, w, h = CROP
        self._camera_info = {
            "width": w,
            "height": h,
            "K": np.reshape(msg.K, (3, 3)).tolist(),
            "D": msg.D,
            "P": np.reshape(msg.P, (3, 4)).tolist(),
            # TODO: this should come from ROS as well
            "H": H_static,
        }
        _K = self._camera_info["K"]
        _K[0][2] = _K[0][2] - x
        _K[1][2] = _K[1][2] - y
        # update P
        _P = self._camera_info["P"]
        _P[0][2] = _P[0][2] - x
        _P[1][2] = _P[1][2] - y
        self._queue_put("camera_info", self._camera_info)

    def _line_detector(self):
        color_ranges: Dict[str, ColorRange] = {
            "white": ColorRange.fromDict({
                "low": [0, 0, 150],
                "high": [180, 100, 255]
            }),
            "yellow": ColorRange.fromDict({
                "low": [25, 140, 100],
                "high": [45, 255, 255]
            })
        }
        color_order = ["yellow", "white"]
        colors_to_detect = [color_ranges[c] for c in color_order]
        detector = LineDetector()
        while not self.is_shutdown:
            bgr = self._queue_pop("image")
            color_detections: List[Detections] = detector.detect(bgr, colors_to_detect)
            lines: Dict[str, dict] = {}
            for i, detections in enumerate(color_detections):
                color = color_order[i]
                # pack detections in a dictionary
                lines[color] = {
                    "lines": detections.lines.tolist(),
                    "centers": detections.centers.tolist(),
                    "normals": detections.normals.tolist(),
                    "color": color_ranges[color].representative
                }
            self._queue_put("lines", lines)

    def _lane_filter(self):
        self._queue_pop("camera_info")
        camera = CameraModel(
            width=self._camera_info["width"],
            height=self._camera_info["height"],
            K=self._camera_info["K"],
            D=self._camera_info["D"],
            P=self._camera_info["P"],
            H=self._camera_info["H"],
        )
        projector = GroundProjector(camera)
        # create filter
        filter = LaneFilterHistogram()
        while not self.is_shutdown:
            segments: List[Segment] = []
            lines = self._queue_pop("lines")
            if self._queue_full("segments"):
                continue
            for color, colored_lines in lines.items():
                # grounded_segments[color] = []
                for line in colored_lines["lines"][:40]:
                    # distorted pixels
                    p0: Pixel = Pixel(line[0], line[1])
                    p1: Pixel = Pixel(line[2], line[3])
                    # distorted pixels to rectified pixels
                    p0_rect: Pixel = camera.rectifier.rectify_pixel(p0)
                    p1_rect: Pixel = camera.rectifier.rectify_pixel(p1)
                    # rectified pixel to normalized coordinates
                    p0_norm: NormalizedImagePoint = camera.pixel2vector(p0_rect)
                    p1_norm: NormalizedImagePoint = camera.pixel2vector(p1_rect)
                    # project image point onto the ground plane
                    grounded_p0: GroundPoint = projector.vector2ground(p0_norm)
                    grounded_p1: GroundPoint = projector.vector2ground(p1_norm)
                    # add grounded segment to output
                    segments.append(Segment(
                        points=[grounded_p0, grounded_p1],
                        color=SegmentColor.WHITE
                    ))
            # apply update
            filter.update(segments)
            # predict
            # if last_update is not None:
            #     delta_t: float = time.time() - last_update
            #     v, w = self._queue_pop("v_w_executed")
            #     filter.predict(delta_t, v, w)
            # get new estimate
            d_hat, phi_hat = filter.get_estimate()
            print(d_hat, phi_hat)
            self._queue_put("d_phi", (d_hat, phi_hat))

    def _lane_controller(self):
        controller = PIDLaneController()
        while not self.is_shutdown:
            d_hat, phi_hat = self._queue_pop("d_phi")
            controller.update(d_hat, phi_hat, time.time())
            v, w = controller.compute_commands()
            self._queue_put("v_w", (v, w))

    def _inverse_kinematics(self):
        ik = InverseKinematics(
            wheel_baseline=wheel_baseline,
            wheel_radius=wheel_radius,
            v_max=v_max,
            omega_max=omega_max,
            # not needed
            gain=-1,
            trim=-1,
            k=-1,
            limit=-1,
        )
        while not self.is_shutdown:
            v, w = self._queue_pop("v_w")
            wl, wr = ik.get_wheels_speed(v, w)
            self._queue_put("v_w_executed", (v, w))
            self._queue_put("wl_wr", (wl, wr))

    def _wheels(self):
        ik = InverseKinematics(
            gain=gain,
            trim=trim,
            k=k,
            limit=limit,
            # not needed
            wheel_baseline=-1,
            wheel_radius=-1,
            v_max=-1,
            omega_max=-1,
        )
        while not self.is_shutdown:
            wl, wr = self._queue_pop("wl_wr")
            dc_l, dc_r = ik.get_wheels_duty_cycle_from_wheels_speed(wl, wr)
            self._wheels_pub.publish(WheelsCmdStamped(
                vel_left=dc_l,
                vel_right=dc_r,
            ))


if __name__ == "__main__":
    node = LaneFollowing()
    # spin forever
    rospy.spin()
