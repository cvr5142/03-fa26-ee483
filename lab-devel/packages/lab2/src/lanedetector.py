#!/usr/bin/env python3

import rospy
import os
import cv2
import numpy as np
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge

class LaneDetecor:
    def __init__(self):
        self.bridge = CvBridge()
        
        self.crop_pub = rospy.Publisher("image_cropped",Image, queue_size=1)
        self.white_pub = rospy.Publisher("image_white",Image, queue_size=1)
        self.yellow_pub = rospy.Publisher("image_yellow",Image, queue_size=1)
        self.edges_pub = rospy.Publisher("image_edges",Image, queue_size=1)
        self.lines_pub = rospy.Publisher("image_lines",Image, queue_size=1)

        self.sub = rospy.Subscriber("/ee403mm03/camera_node/image/compressed", CompressedImage,self.image_cb, queue_size=1, buff_size = 10000000)

    def image_cb(self,msg):
        #Convert compressed image
        image = self.bridge.compressed_imgmsg_to_cv2(msg)

        height = image.shape[0]
        cropped = image[int(height*0.5):,:]

        #Convert to HSV
        hsv = cv2.cvtColor(cropped,cv2.COLOR_BGR2HSV)

        lower_white=np.array(rospy.get_param("lower_white",[0,0,170],dtype=np.uint8))
        upper_white=np.array(rospy.get_param("upper_white",[180,70,255],dtype=np.uint8))
        white_mask=cv2.inRange(hsv,lower_white,upper_white)

        lower_yellow=np.array(rospy.get_param("lower_yellow",[15,80,80],dtype=np.uint8))
        upper_yellow=np.array(rospy.get_param("uoer_yellow",[40,255,255],dtype=np.uint8))
        yellow_mask = cv2.inRange(hsv, lower_yellow,upper_yellow)

        #edge detection
        gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray,50,150)

        #Combine edges with Color masks
        white_edges=cv2.bitwise_and(edgs,white_mask)
        yellow_edges=cv2.bitwise_and(edges,yellow_mask)

        #Hough transform
        white_lines = cv2.HoughLinesP(white_edges,1,np.pi/180,threshold=20,minLineLength=20,maxLineGap=10)

        yellow_lines= cv2.HoughLinesP(white_edges,1,np.pi/180,threshold=20,minLineLength=20,maxLineGap=10)




