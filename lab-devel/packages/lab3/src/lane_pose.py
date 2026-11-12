#!/usr/bin/env python3

import os
import threading
import numpy as np
import rospy
import yaml
from duckietown_msgs.msg import LanePose, SegmentList, Twist2DStamped, WheelEncoderStamped
from sensor_msgs.msg import CameraInfo

from lane_pose_kf import GroundProjector, LanePoseKF

class PIDController:
    def __init__(self):
        self.integral = 0
        self.prev_error = 0
        self.prev_time = None
        #self.params = rospy.get_param("~")
        self.kp = rospy.get_param("/kp",0)
        self.ki = rospy.get_param("/ki",0)
        self.kd = rospy.get_param("/kd",0)

    def calculate(self,error):
        self.kp = rospy.get_param("/kp",0)
        self.ki = rospy.get_param("/ki",0)
        self.kd = rospy.get_param("/kd",0)
        print(self.kp)

        current_time=rospy.get_time()
        if self.prev_time is None:
            dt = 0
        else:
            dt = current_time - self.prev_time

        P = self.kp* error
        
        if dt>0:
            self.integral += error * dt + self.kd*(error-self.prev_error) /dt
            D=self.kd * (error -self.prev_error)/dt
        else:
            D = 0

        I = self.ki* self.integral
        angular_velocity = P+I+D
        print(angular_velocity)

        self.prev_error = error
        self.prev_time = current_time

        return angular_velocity
        
    # def pid_moving(self, msg):
    #     error = - msg.phi
    #     angular_velocity = self.controller.calculate(error)

    #     car_cmd = Twist2DStamped()
    #     car_cmd.v = 0.2
    #     car_cmd.omega = max(-1.0, min(1.0, angular_velocity))

    #     self.control_pub.publish(car_cmd)
        #D = msg.d

        #if self.prev_time is None:
         #   dt = 0
        #else:
         #   dt = current_time - self.prev_time
        
        # P = kp * error
        # D = 0
        # if dt>0:
        #     self.integral += error*dt
        #     D = kd*(error-self.prev_error)/dt
        
        # I = ki * self.integral
        # angular_velocity = P + I + D

        # self.prev_error = error
        # self.prev_time = current Time

        # return angular_velocity





class PidNode:
    def __init__(self):
        self.controller = PIDController()
        self.control_pub = rospy.Publisher("car_cmd_switch_node/cmd", Twist2DStamped, queue_size=10)
        # self.error_sub = rospy.Subscriber("error", Float32, self.error_callback)
        rospy.Subscriber("/ee483mm03/lane_filter_node/lane_pose", LanePose, self.pid_moving, queue_size=10)
        
        rospy.sleep(0.5)
        # rospy.set_param("controller_ready","true")

    def pid_moving(self, msg):
        print("Yay")
        error = - msg.phi
        angular_velocity = self.controller.calculate(error)

        car_cmd = Twist2DStamped()
        car_cmd.v = 0
        car_cmd.omega = angular_velocity

        self.control_pub.publish(car_cmd)

    def error_callback(self,msg):
        error=msg.data
        angular_velocity=self.controller.calculate(error)
        self.control_pub.publish(angular_velocity)

if __name__ == "__main__":
    rospy.init_node("pid_controller")
    node=PidNode()
    rospy.spin()
    
        


