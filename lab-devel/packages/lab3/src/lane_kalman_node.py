#!/usr/bin/env python3
"""
Lane pose (d, phi) from YOUR line segments, with a Kalman filter. Replaces dt-core's
ground_projection + lane_filter nodes and publishes the same message (duckietown_msgs/LanePose).

Subscribes:
    /<robot>/line_detector_node/segment_list         SegmentList, your line_detection.py output
                                                     (pixels_normalized; endpoint order does not matter)
    /<robot>/camera_node/camera_info                 CameraInfo, once, for the intrinsic calibration
    /<robot>/left_wheel_encoder_node/tick            WheelEncoderStamped  \  odometry for the
    /<robot>/right_wheel_encoder_node/tick           WheelEncoderStamped   > predict step
    /<robot>/car_cmd_switch_node/cmd                 Twist2DStamped       /  (cmd if encoders stall)
Publishes:
    ~lane_pose            LanePose: Kalman estimate (d, phi, in_lane; d_phi_covariance = its uncertainty)
    ~lane_pose_measured   LanePose: this frame's measurement only (no filtering), for comparison

With the node named lane_kalman_node, ~lane_pose is /<robot>/lane_kalman_node/lane_pose.
Point your lane_controller.py at that topic (instead of lane_filter_node/lane_pose).

Each time a segment list arrives:
    1. odometry since the last list: ds (distance) and dtheta (heading change) from the encoders
    2. project the segments to the ground (meters, robot frame) with the camera calibration
    3. fit the white and yellow stripes -> one measurement (d, phi)
    4. Kalman predict with (ds, dtheta), update with the measurement -> publish
The math is in lane_pose_kf.py; the parameters are in config/lane_kalman_node.yaml.
"""
import os
import threading

import numpy as np
import rospy
import yaml
from duckietown_msgs.msg import LanePose, SegmentList, Twist2DStamped, WheelEncoderStamped
from sensor_msgs.msg import CameraInfo

from lane_pose_kf import GroundProjector, LanePoseKF

CALIB_DIR = "/data/config/calibrations"


def load_calibration_yaml(kind, robot):
    """/data/config/calibrations/<kind>/<robot>.yaml, or default.yaml, or None."""
    for name in (robot, "default"):
        path = os.path.join(CALIB_DIR, kind, name + ".yaml")
        if os.path.isfile(path):
            with open(path) as f:
                rospy.loginfo(f"[lane_kalman_node] {kind}: {path}")
                return yaml.safe_load(f)
    return None


class LaneKalmanNode:
    def __init__(self):
        robot = os.environ["VEHICLE_NAME"]
        self.params = rospy.get_param("~")  # everything from config/lane_kalman_node.yaml
        self.use_ground = bool(self.params.get("use_ground_projection", False))
        self.encoder_timeout = float(self.params.get("encoder_timeout", 0.5))
        self.in_lane_timeout = float(self.params.get("in_lane_timeout", 1.0))
        self.in_lane_max_sigma_phi = float(self.params.get("in_lane_max_sigma_phi", 0.5))

        # wheel geometry for the encoders
        kin = load_calibration_yaml("kinematics", robot) or {}
        self.radius = float(kin.get("radius", 0.0318))
        self.baseline = float(kin.get("baseline", 0.1))

        projector = None if self.use_ground else self.make_projector(robot)
        self.estimator = LanePoseKF(self.params, projector)

        # odometry state, written by the encoder/cmd callbacks, read by the segment callback
        self.lock = threading.Lock()
        self.ticks = {"left": None, "right": None}  # (time, ticks, resolution) of the last message
        self.ticks_at_last_update = None
        self.cmd = (0.0, 0.0)
        self.last_time = None
        self.last_accepted = None

        self.pub = rospy.Publisher("~lane_pose", LanePose, queue_size=1)
        self.pub_meas = rospy.Publisher("~lane_pose_measured", LanePose, queue_size=1)
        rospy.Subscriber(f"/{robot}/left_wheel_encoder_node/tick", WheelEncoderStamped,
                         self.cb_encoder, callback_args="left", queue_size=10)
        rospy.Subscriber(f"/{robot}/right_wheel_encoder_node/tick", WheelEncoderStamped,
                         self.cb_encoder, callback_args="right", queue_size=10)
        rospy.Subscriber(f"/{robot}/car_cmd_switch_node/cmd", Twist2DStamped, self.cb_cmd, queue_size=1)
        seg_topic = (f"/{robot}/ground_projection_node/lineseglist_out" if self.use_ground
                     else f"/{robot}/line_detector_node/segment_list")
        rospy.Subscriber(seg_topic, SegmentList, self.cb_segments, queue_size=1)
        rospy.loginfo(f"[lane_kalman_node] listening to {seg_topic}")

    def make_projector(self, robot):
        # intrinsics: from the camera driver (same numbers as the intrinsic calibration file)
        try:
            ci = rospy.wait_for_message(f"/{robot}/camera_node/camera_info", CameraInfo, timeout=15)
            K, D, R, P, w, h = ci.K, ci.D, ci.R, ci.P, ci.width, ci.height
        except rospy.ROSException:
            cam = load_calibration_yaml("camera_intrinsic", robot)
            if cam is None:
                rospy.signal_shutdown("no camera_info and no intrinsic calibration file")
                raise
            K, D = cam["camera_matrix"]["data"], cam["distortion_coefficients"]["data"]
            R, P = cam["rectification_matrix"]["data"], cam["projection_matrix"]["data"]
            w, h = cam["image_width"], cam["image_height"]
        # extrinsics: the launch file's homography, or the robot's calibration file
        H = self.params.get("homography") or []
        if len(H) != 9:
            ext = load_calibration_yaml("camera_extrinsic", robot)
            if ext is None:
                rospy.signal_shutdown("no extrinsic calibration (homography)")
                raise RuntimeError("no homography")
            H = ext["homography"]
        else:
            rospy.loginfo("[lane_kalman_node] camera_extrinsic: homography from the parameters")
        return GroundProjector(K, D, R, P, w, h, H)

    # ---- odometry -----------------------------------------------------------------------------
    def cb_encoder(self, msg, side):
        with self.lock:
            self.ticks[side] = (rospy.get_time(), msg.data, msg.resolution)

    def cb_cmd(self, msg):
        with self.lock:
            self.cmd = (msg.v, msg.omega)

    def odometry(self, now, dt):
        """(ds, dtheta) since the last segment list: encoders, or car_cmd if they stalled."""
        with self.lock:
            left, right = self.ticks["left"], self.ticks["right"]
            fresh = (left is not None and right is not None
                     and now - min(left[0], right[0]) < self.encoder_timeout)
            if fresh:
                current = (left[1], right[1])
                previous = self.ticks_at_last_update or current
                self.ticks_at_last_update = current
                m_per_tick = 2 * np.pi * self.radius / left[2]
                dl = (current[0] - previous[0]) * m_per_tick
                dr = (current[1] - previous[1]) * m_per_tick
                return (dl + dr) / 2, (dr - dl) / self.baseline
            self.ticks_at_last_update = None
            v, omega = self.cmd
            return v * dt, omega * dt

    # ---- main loop: one segment list -> one lane pose -------------------------------------------
    def cb_segments(self, msg):
        now = rospy.get_time()
        dt = 0.0 if self.last_time is None else now - self.last_time
        self.last_time = now
        ds, dtheta = self.odometry(now, dt)

        pts = [[[p.x, p.y] for p in (s.points if self.use_ground else s.pixels_normalized)]
               for s in msg.segments]
        colors = [s.color for s in msg.segments]
        r = self.estimator.step(np.array(pts, float).reshape(-1, 2, 2), np.array(colors, int),
                                ds, dtheta, dt, on_ground=self.use_ground)

        if r["status"] in ("accepted", "init", "restart"):
            self.last_accepted = now
        stamp = msg.header.stamp if msg.header.stamp.to_sec() > 0 else rospy.Time.now()
        if np.isfinite(r["d_meas"]):
            self.pub_meas.publish(self.lane_pose(stamp, r["d_meas"], r["phi_meas"], None, True))
        if np.isfinite(r["d"]):
            in_lane = (self.last_accepted is not None and now - self.last_accepted < self.in_lane_timeout
                       and np.sqrt(r["P"][1, 1]) < self.in_lane_max_sigma_phi)
            self.pub.publish(self.lane_pose(stamp, r["d"], r["phi"], r["P"], in_lane))

    @staticmethod
    def lane_pose(stamp, d, phi, P, in_lane):
        msg = LanePose()
        msg.header.stamp = stamp
        msg.d, msg.phi = float(d), float(phi)
        if P is not None:  # [var_d, cov, cov, var_phi]; sigma_d = sqrt(var_d), sigma_phi = sqrt(var_phi)
            msg.d_phi_covariance = [float(v) for v in P.ravel()]
        msg.in_lane = bool(in_lane)
        msg.status = LanePose.NORMAL
        return msg


if __name__ == "__main__":
    rospy.init_node("lane_kalman_node")
    LaneKalmanNode()
    rospy.spin()
