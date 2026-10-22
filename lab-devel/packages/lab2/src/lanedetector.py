#!/usr/bin/env python3

import rospy
import os
import cv2
import numpy as np
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge

class LaneDetector:
    def __init__(self):
        self.bridge = CvBridge()
        
        self.crop_pub = rospy.Publisher("image_cropped",Image, queue_size=1)
        self.white_pub = rospy.Publisher("image_white",Image, queue_size=1)
        self.yellow_pub = rospy.Publisher("image_yellow",Image, queue_size=1)
        self.white_edges_pub = rospy.Publisher("image_white_edges",Image, queue_size=1)
        self.yellow_edges_pub = rospy.Publisher("image__yellow_edges",Image, queue_size=1)
        
        self.lines_pub = rospy.Publisher("image_lines",Image, queue_size=1)
        

        rospy.Subscriber("/ee483mm03/camera_node/image/compressed", CompressedImage,self.image_cb, queue_size=1, buff_size = 10000000)

    def image_cb(self,msg):
        #Convert compressed image
        image = self.bridge.compressed_imgmsg_to_cv2(msg)

        height = image.shape[0]
        cropped = image[int(height*0.5):,:]
        cropped_imgmsg = self.bridge.cv2_to_imgmsg(cropped, "bgr8")
        self.crop_pub.publish(cropped_imgmsg)

        #Convert to HSV FIX THE MASKS THEY AREN'T DETECTING ENOUGH
        hsv = cv2.cvtColor(cropped,cv2.COLOR_BGR2HSV)

        white_mask=cv2.inRange(hsv,(0,0,90),(180,70,255))
        white_mask_combined = cv2.bitwise_and(cropped,cropped,mask=white_mask)
        white_mask_imgmsg = self.bridge.cv2_to_imgmsg(white_mask_combined, "bgr8")
        
        self.white_pub.publish(white_mask_imgmsg)

        yellow_mask = cv2.inRange(hsv, (15,70,80),(40,255,255))
        yellow_mask_combined = cv2.bitwise_and(cropped,cropped,mask=yellow_mask)
        yellow_mask_imgmsg = self.bridge.cv2_to_imgmsg(yellow_mask_combined, "bgr8")
        self.yellow_pub.publish(yellow_mask_imgmsg)

        #edge detection
        gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray,30,100)

        #Combine edges with Color masks
        white_edges=cv2.bitwise_and(edges,white_mask)
        white_edges_imgmsg = self.bridge.cv2_to_imgmsg(white_edges,"mono8")
        self.white_edges_pub.publish(white_edges_imgmsg)

        yellow_edges=cv2.bitwise_and(edges,yellow_mask)
        yellow_edges_imgmsg = self.bridge.cv2_to_imgmsg(yellow_edges,"mono8")
        self.yellow_edges_pub.publish(yellow_edges_imgmsg)

        #Hough transform
        white_lines = cv2.HoughLinesP(white_edges,1,np.pi/180,threshold=10,minLineLength=10,maxLineGap=10)
        yellow_lines= cv2.HoughLinesP(yellow_edges,1,np.pi/180,threshold=10,minLineLength=10,maxLineGap=10)

        lined_white = self.output_lines(cropped,white_lines)
        lined_full = self.output_lines(lined_white,yellow_lines)
        lined_full_imgmsg = self.bridge.cv2_to_imgmsg(lined_full,"bgr8")
        self.lines_pub.publish(lined_full_imgmsg)


    def output_lines(self, original_img, lines):
        output = np.copy(original_img)
        if lines is not None:
            for i in range(len(lines)):
                l = lines[i][0]
                cv2.line(output, (l[0],l[1]), (l[2],l[3]), (255,0,0), 2, cv2.LINE_AA)
                cv2.circle(output, (l[0],l[1]),2,(0,255,0))
                cv2.circle(output, (l[2],l[3]),2,(0,0,255))
        return output

if __name__=="__main__":
    # initialize our node and create a publisher as normal
    rospy.init_node("lane_detector", anonymous=True)
    lane_detector = LaneDetector()
    rospy.spin()

