#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image , CompressedImage
from cv_bridge import CvBridge
import cv2
import numpy as np
import math
import pyocr
import pyocr.builders
from PIL import Image

tools = pyocr.get_available_tools()
tool = tools[0]

p1 = np.array([520, 311])
p2 = np.array([682, 310])
p3 = np.array([498, 408])
p4 = np.array([685, 408])

src = np.float32([p1, p2, p3, p4])
dst = np.float32([[0,0],[100,0],[0,100],[100,100]])


M = cv2.getPerspectiveTransform(src, dst)
kernel = np.ones((3, 3), np.uint8)

def process_image(msg):
    try:
        bridge = CvBridge()
        orig = bridge.compressed_imgmsg_to_cv2(msg, "bgr8")

 	hsvLower_y = np.array([0, 64, 10])    
 	hsvUpper_y = np.array([100, 240 ,240])   
     
 	hsvLower_m = np.array([110, 0, 0])   
 	hsvUpper_m = np.array([140, 240,240])  
     
 	hsvLower_c = np.array([150, 64, 10])    
 	hsvUpper_c = np.array([180, 240 ,240])

        hsv = cv2.cvtColor(orig, cv2.COLOR_BGR2HSV)

        hsv_mask_y = cv2.inRange(orig, hsvLower_y, hsvUpper_y)
 	hsv_mask_m = cv2.inRange(orig, hsvLower_m, hsvUpper_m)
 	hsv_mask_c = cv2.inRange(orig, hsvLower_c, hsvUpper_c)
	
	ret, img_y = cv2.threshold(hsv_mask_y, 100, 255, cv2.THRESH_BINARY)
        img_y = cv2.erode(hsv_mask_y,kernel,iterations = 3)
	ret, img_c= cv2.threshold(hsv_mask_c, 100, 255, cv2.THRESH_BINARY)
        img_c = cv2.erode(hsv_mask_c,kernel,iterations = 3)
	ret, img_m= cv2.threshold(hsv_mask_m, 100, 255, cv2.THRESH_BINARY)
        img_m = cv2.erode(hsv_mask_m,kernel,iterations = 3)	
	
	moji_y = cv2.warpPerspective(img_y, M,(100,100))
	moji_c = cv2.warpPerspective(img_c, M,(100,100))
	moji_m = cv2.warpPerspective(img_m, M,(100,100))

	moji90_y = cv2.rotate(moji_y, cv2.ROTATE_90_CLOCKWISE)
	moji90_m = cv2.rotate(moji_m, cv2.ROTATE_90_CLOCKWISE)
	moji90_c = cv2.rotate(moji_c, cv2.ROTATE_90_CLOCKWISE)

	img = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
        ret, img = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)
        img = cv2.erode(img,kernel,iterations = 3)
        moji = cv2.warpPerspective(img, M,(100,100))
        moji90 = cv2.rotate(moji, cv2.ROTATE_90_CLOCKWISE)
        #moji90 =mojiz
        pil_img = Image.fromarray(moji90)
	
	pil_img_y = Image.fromarray(moji90_y)
	pil_img_m = Image.fromarray(moji90_m)
	pil_img_c = Image.fromarray(moji90_c)	

        cv2.imshow('Y', moji_y)
	cv2.imshow('C', moji_c)
	cv2.imshow('M', moji_m)

	cv2.imshow('Y2', moji_y)
	cv2.imshow('C2', moji_c)
	cv2.imshow('M2', moji_m)
        #cv2.imshow('image', img)

        #cv2.imshow('m',hsv_mask_m)
 	#cv2.imshow('y',hsv_mask_y)
 	#cv2.imshow('c',hsv_mask_c)

        builder = pyocr.builders.TextBuilder(tesseract_layout=5)
        result = tool.image_to_string(pil_img, lang="eng", builder=builder)

        if 'c' in result:
            result = 'C'
        elif 'C' in result:
            result = 'C'
        elif 'E' in result:
            result = 'B'
        else:
            result='A'

        print(result)
        cv2.waitKey(1)
    except Exception as err:
        print err

def start_node():
    rospy.init_node('img_proc')
    rospy.loginfo('img_proc node started')
    rospy.Subscriber("/camera/color/image_raw/compressed", CompressedImage, process_image)
    rospy.spin()

if __name__ == '__main__':
    try:
        start_node()
    except rospy.ROSInterruptException:
        pass
