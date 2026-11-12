#!/usr/bin/env python3
import sys
import rospy
import cv2
import matplotlib
import numpy as np
import os
from pathlib import Path
from message_filters import ApproximateTimeSynchronizer, Subscriber
from duckietown_msgs.msg import Segment, SegmentList, Vector2D
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge
from std_srvs.srv import SetBool, SetBoolResponse

# Logistic regression (Ex5) weights, used when ~mask is "lr"
DEFAULT_LR_WEIGHTS = str(Path(__file__).resolve().parent.parent / "models" / "model_weights.npz")



class ImageProcessor: 
    def __init__(self):
        # processor() ignores images until the end of __init__ (callbacks run in another thread)
        self.ready = False
        # Instatiate the converter class once by using a class member
        rospy.Service('line_detector_node/switch', SetBool, self.ld_switch)
        rospy.Service('lane_filter_node/switch',SetBool, self.lf_switch)
        self.bridge = CvBridge()
        veh_name = os.environ['VEHICLE_NAME']
        self.pub_edges = rospy.Publisher("image_edges", Image, queue_size=10)
        self.pub_yellow_mask = rospy.Publisher("image_mask_yellow", Image, queue_size=10)
        self.pub_white_mask = rospy.Publisher("image_mask_white", Image, queue_size=10)
        self.pub_yellow_lines = rospy.Publisher("image_lines_yellow", Image, queue_size=10)
        self.pub_white_lines = rospy.Publisher("image_lines_white", Image, queue_size=10)
        self.pub_lines = rospy.Publisher("image_lines", Image, queue_size=10)
        self.pub_segments = rospy.Publisher("line_detector_node/segment_list", SegmentList, queue_size=10)
        
        # Color filter for the lane masks (launch file parameter ~mask):
        #   "hsv": cv2.inRange boxes in white_filter / yellow_filter (your values)
        #   "lr":  the Ex5 logistic regression classifier (ml_filter.py + models/model_weights.npz)
        self.mask_method = rospy.get_param("~mask", "hsv")
        if self.mask_method == "lr":
            from ml_filter import ColorClassifier
            self.classifier = ColorClassifier(rospy.get_param("~lr_weights", DEFAULT_LR_WEIGHTS))
            self.lr_min_confidence = {"White": rospy.get_param("~lr_min_confidence_white", 0.5),
                                      "Yellow": rospy.get_param("~lr_min_confidence_yellow", 0.5)}
            self.lr_dilation = int(rospy.get_param("~lr_dilation", 5))
        elif self.mask_method != "hsv":
            raise ValueError(f"~mask must be 'hsv' or 'lr', not {self.mask_method!r}")
        rospy.loginfo(f"[line_detection] color masks: {self.mask_method} ({__file__})")
        self.lr_masks = None

        # Your Lab 2 values, set in lab3.launch (defaults below are used if a parameter is missing)
        self.white_hsv = (tuple(rospy.get_param("~white_hsv_lower", [70, 0, 145])),
                          tuple(rospy.get_param("~white_hsv_upper", [140, 120, 255])))
        self.yellow_hsv = (tuple(rospy.get_param("~yellow_hsv_lower", [20, 33, 100])),
                           tuple(rospy.get_param("~yellow_hsv_upper", [86, 176, 222])))
        self.canny = rospy.get_param("~canny", [50, 150])
        self.hough_white = rospy.get_param("~hough_white", {"threshold": 10, "min_line_length": 20, "max_line_gap": 3})
        self.hough_yellow = rospy.get_param("~hough_yellow", {"threshold": 0, "min_line_length": 20, "max_line_gap": 10})

        #variables used in class
        self.hsv_image = None
        self.cv_img = None
        self.cv_crop = None
        self.white_mask = None
        self.yellow_mask = None

        # Subscribe last: images can arrive as soon as this line runs, and processor() needs
        # everything above (mask method, LR classifier, HSV values) to exist already
        rospy.Subscriber(f"/{veh_name}/camera_node/image/compressed", CompressedImage, self.processor, queue_size=1, buff_size= 2**24)
        #rospy.Subscriber("image", Image, self.processor, queue_size=1, buff_size= 2**24)
        self.ready = True

    def processor(self, msg):
        if not self.ready:  # __init__ has not finished yet
            return
        self.cv_img = self.bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
        #self.cv_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        self.crop() #Crop Image
        self.hsv_image = cv2.cvtColor(self.cv_crop, cv2.COLOR_BGR2HSV)
        if self.mask_method == "lr":
            # one classifier pass gives the White and Yellow masks (0/255, same size as the crop)
            self.lr_masks = self.classifier.segment(self.cv_crop, min_confidence=self.lr_min_confidence)
        self.white_filter() #Create white filter
        self.yellow_filter() #create yellow filter
        self.segment() #Create segments from filters

    # DO NOT CHANGE THIS FUNCTION
    def crop(self):
        image_size = (160, 120)
        offset = 40
        new_image = cv2.resize(self.cv_img, image_size, interpolation=cv2.INTER_NEAREST)
        self.cv_crop= new_image[offset:, :]

    def lr_mask(self, name):
        # Dilate the classifier mask so it also covers the Canny edges on the stripe border
        # (lr_dilation in lab3.launch: kernel size in pixels, 0 or 1 = no dilation)
        if self.lr_dilation <= 1:
            return self.lr_masks[name]
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.lr_dilation, self.lr_dilation))
        return cv2.dilate(self.lr_masks[name], kernel)

    def white_filter(self):
        if self.mask_method == "lr":
            self.white_mask = self.lr_mask("White")
            self.pub_white_mask.publish(self.bridge.cv2_to_imgmsg(self.white_mask, "mono8"))
            return
        #Finds all pixels within the given range (white_hsv_lower/upper in lab3.launch)
        mask = cv2.inRange(self.hsv_image, *self.white_hsv)
        # mask2 = cv2.inRange(self.hsv_image, (13,0,154),(33,255,255))
        # mask = cv2.bitwise_or(mask,mask2)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        # image_erode = cv2.erode(red_part, kernel)
        image_erode = cv2.erode(mask, kernel)
        image_dilate = cv2.dilate(image_erode, kernel2)
        self.white_mask = image_dilate
        #Converting image from cv format to ROS msg using 
        white_mask = self.bridge.cv2_to_imgmsg(self.white_mask, "mono8")
        #publishing cropped image
        self.pub_white_mask.publish(white_mask)


    def yellow_filter(self):
        if self.mask_method == "lr":
            self.yellow_mask = self.lr_mask("Yellow")
            self.pub_yellow_mask.publish(self.bridge.cv2_to_imgmsg(self.yellow_mask, "mono8"))
            return
        #Finds all pixels within the given range (yellow_hsv_lower/upper in lab3.launch)
        mask = cv2.inRange(self.hsv_image, *self.yellow_hsv)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        # image_erode = cv2.erode(red_part, kernel)
        image_erode = cv2.erode(mask, kernel)
        image_dilate = cv2.dilate(image_erode, kernel2)
        self.yellow_mask = image_dilate
        #Converting image from cv format to ROS msg using 
        white_mask = self.bridge.cv2_to_imgmsg(self.yellow_mask, "mono8")
        #publishing cropped image

        self.pub_yellow_mask.publish(white_mask)

    def segment(self):
        # Canny thresholds: canny in lab3.launch
        edges = cv2.Canny(self.cv_crop, *self.canny)

        #Combinining edges with mask white 
        edges_white = cv2.bitwise_and(edges,self.white_mask)
        ros_edges = self.bridge.cv2_to_imgmsg(edges_white, "mono8")
        self.pub_white_lines.publish(ros_edges)

        #Combinining edges with mask yellow 
        edges_yellow = cv2.bitwise_and(edges,self.yellow_mask)
        ros_edges = self.bridge.cv2_to_imgmsg(edges_yellow, "mono8")
        self.pub_yellow_lines.publish(ros_edges)

        
        #Finding Line Segments (hough_white / hough_yellow in lab3.launch)
        hough_white = self.hough(edges_white, self.hough_white)
        hough_yellow = self.hough(edges_yellow, self.hough_yellow)

        # DO NOT CHANGE
        #Publish and  Segments
        final_lines = self.output_lines(self.cv_crop,hough_white,hough_yellow)
        ros_lines = self.bridge.cv2_to_imgmsg(final_lines, "bgr8")
        self.pub_lines.publish(ros_lines)
        self.convert_lines(hough_white,hough_yellow)


    @staticmethod
    def hough(edges, p):
        return cv2.HoughLinesP(edges, 1, np.pi / 180, int(p["threshold"]),
                               minLineLength=int(p["min_line_length"]), maxLineGap=int(p["max_line_gap"]))

    def convert_lines(self,lines_white,lines_yellow):
        if lines_white is not None:
            lines_reshaped =lines_white.reshape((-1,4))
        else:
            lines_reshaped =[]
        if lines_yellow is not None:
            lines_reshaped_yellow =lines_yellow.reshape((-1,4))
        else:
            lines_reshaped_yellow =[]
        offset = 40
        image_size = (160,120)
        arr_cutoff = np.array([0,offset, 0, offset])
        arr_ratio = np.array([([1. /image_size[0], 1. /image_size[1], 1. /image_size[0], 1. /image_size[1]])])
        if lines_white is not None:
            line_normalized_white = (lines_reshaped + arr_cutoff) * arr_ratio
        else:
            line_normalized_white= np.empty(shape= [0,0,])
            
        if lines_yellow is not None:
            line_normalized_yellow = (lines_reshaped_yellow + arr_cutoff) * arr_ratio
        else:
            line_normalized_yellow= np.empty(shape= [0,0,])
        
        Line_List = SegmentList()
        Line_Array = []
        # print(line_normalized_white)
        for i in range(line_normalized_white.shape[0]):
        # for i in range(1):

            line = Segment()
            line.color = line.WHITE
            Initial_Line = Vector2D(line_normalized_white[i,0], line_normalized_white[i,1] )
            End_Line = Vector2D(line_normalized_white[i,2], line_normalized_white[i,3] )
            line.pixels_normalized[0] = Initial_Line
            line.pixels_normalized[1] = End_Line
            Line_Array.append(line)
        for i in range(line_normalized_yellow.shape[0]):
        # for i in range(1):

            line = Segment()
            line.color = line.YELLOW
            Initial_Line = Vector2D(line_normalized_yellow[i,0], line_normalized_yellow[i,1] )
            End_Line = Vector2D(line_normalized_yellow[i,2], line_normalized_yellow[i,3] )
            line.pixels_normalized[0] = Initial_Line
            line.pixels_normalized[1] = End_Line
            Line_Array.append(line)
        Line_List.segments= Line_Array
        Line_List.header.stamp = rospy.Time.now()
        self.pub_segments.publish(Line_List)


    def output_lines(self, original_image, lines,lines_yellow):
        output = np.copy(original_image)
        if lines is not None:
            for i in range(len(lines)):
                l = lines[i][0]
                cv2.line(output, (l[0],l[1]), (l[2],l[3]), (255,0,0), 2, cv2.LINE_AA)
                cv2.circle(output, (l[0],l[1]), 2, (0,255,0))
                cv2.circle(output, (l[2],l[3]), 2, (0,0,255))
        if lines_yellow is not None:
            for i in range(len(lines_yellow)):
                l = lines_yellow[i][0]
                cv2.line(output, (l[0],l[1]), (l[2],l[3]), (0,0,255), 2, cv2.LINE_AA)
                cv2.circle(output, (l[0],l[1]), 2, (0,255,0))
                cv2.circle(output, (l[2],l[3]), 2, (0,255,255))
        return output
       
       
    
    def ld_switch(self,msg):
        return True,""
    
    def lf_switch(self,msg):
        return True,""




        

if __name__=="__main__":
    # initialize our node and create a publisher as normal
    rospy.init_node("ImageProcessor", anonymous=True)
    ImageProcessor()
    rospy.spin()