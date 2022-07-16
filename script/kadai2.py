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

R_p1 = np.array([413, 276])
R_p2 = np.array([569, 271])
R_p3 = np.array([375, 369])
R_p4 = np.array([557, 365])

L_p1 = np.array([621, 274])
L_p2 = np.array([776, 272])
L_p3 = np.array([615, 365])
L_p4 = np.array([850, 400])

R_src = np.float32([R_p1, R_p2, R_p3, R_p4])
L_src = np.float32([L_p1, L_p2, L_p3, L_p4])
dst = np.float32([[0,0],[100,0],[0,100],[100,100]])

R_M = cv2.getPerspectiveTransform(R_src, dst)
L_M = cv2.getPerspectiveTransform(L_src, dst)
kernel = np.ones((3, 3), np.uint8)


def process_image(msg):
    try:
        bridge = CvBridge()
        orig = bridge.compressed_imgmsg_to_cv2(msg, "bgr8")

 	hsvLower_y = np.array([0, 32, 10])    
 	hsvUpper_y = np.array([70, 240 ,240])   
     
 	hsvLower_m = np.array([60, 0, 0])   
 	hsvUpper_m = np.array([100, 240,240])  
     
 	hsvLower_c = np.array([110, 32, 10])    
 	hsvUpper_c = np.array([140, 240 ,240])

        hsv = cv2.cvtColor(orig, cv2.COLOR_BGR2HSV)

        hsv_mask_y = cv2.inRange(orig, hsvLower_y, hsvUpper_y)
 	hsv_mask_m = cv2.inRange(orig, hsvLower_m, hsvUpper_m)
 	hsv_mask_c = cv2.inRange(orig, hsvLower_c, hsvUpper_c)
	
	ret, img_y = cv2.threshold(hsv_mask_y, 100, 255, cv2.THRESH_BINARY)
        img_y = cv2.erode(hsv_mask_y,kernel,iterations = 3)
	ret, img_c = cv2.threshold(hsv_mask_c, 100, 255, cv2.THRESH_BINARY)
        img_c = cv2.erode(hsv_mask_c,kernel,iterations = 3)
	ret, img_m = cv2.threshold(hsv_mask_m, 100, 255, cv2.THRESH_BINARY)
        img_m = cv2.erode(hsv_mask_m,kernel,iterations = 3)	
	
	R_moji_y = cv2.warpPerspective(img_y, R_M,(100,100))
	R_moji_c = cv2.warpPerspective(img_c, R_M,(100,100))
	R_moji_m = cv2.warpPerspective(img_m, R_M,(100,100))

	L_moji_y = cv2.warpPerspective(img_y, L_M,(100,100))
	L_moji_c = cv2.warpPerspective(img_c, L_M,(100,100))
	L_moji_m = cv2.warpPerspective(img_m, L_M,(100,100))

	R_moji90_y = cv2.rotate(R_moji_y, cv2.ROTATE_90_CLOCKWISE)
	R_moji90_m = cv2.rotate(R_moji_m, cv2.ROTATE_90_CLOCKWISE)
	R_moji90_c = cv2.rotate(R_moji_c, cv2.ROTATE_90_CLOCKWISE)

	L_moji90_y = cv2.rotate(L_moji_y, cv2.ROTATE_90_CLOCKWISE)
	L_moji90_m = cv2.rotate(L_moji_m, cv2.ROTATE_90_CLOCKWISE)
	L_moji90_c = cv2.rotate(L_moji_c, cv2.ROTATE_90_CLOCKWISE)

	R_moji_y_2 = cv2.warpPerspective(img_y, R_M,(40,40))
	R_moji_c_2 = cv2.warpPerspective(img_c, R_M,(40,40))
	R_moji_m_2 = cv2.warpPerspective(img_m, R_M,(40,40))

	L_moji_y_2 = cv2.warpPerspective(img_y, L_M,(40,40))
	L_moji_c_2 = cv2.warpPerspective(img_c, L_M,(40,40))
	L_moji_m_2 = cv2.warpPerspective(img_m, L_M,(40,40))

	R_moji90_y2 = cv2.rotate(R_moji_y_2, cv2.ROTATE_90_CLOCKWISE)
	R_moji90_m2 = cv2.rotate(R_moji_m_2, cv2.ROTATE_90_CLOCKWISE)
	R_moji90_c2 = cv2.rotate(R_moji_c_2, cv2.ROTATE_90_CLOCKWISE)

	L_moji90_y2 = cv2.rotate(L_moji_y_2, cv2.ROTATE_90_CLOCKWISE)
	L_moji90_m2 = cv2.rotate(L_moji_m_2, cv2.ROTATE_90_CLOCKWISE)
	L_moji90_c2 = cv2.rotate(L_moji_c_2, cv2.ROTATE_90_CLOCKWISE)

	#img = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
        #ret, img = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)
        #img = cv2.erode(img,kernel,iterations = 3)
        #moji = cv2.warpPerspective(img, M,(100,100))
        #moji90 = cv2.rotate(moji, cv2.ROTATE_90_CLOCKWISE)
        #moji90 =mojiz
        #pil_img = Image.fromarray(moji90)
	
	#pil_img_y = Image.fromarray(moji90_y)
	#pil_img_m = Image.fromarray(moji90_m)
	#pil_img_c = Image.fromarray(moji90_c)

	#pil_img_y2 = Image.fromarray(moji90_y2)
	#pil_img_m2 = Image.fromarray(moji90_m2)
	#pil_img_c2 = Image.fromarray(moji90_c2)	

	R_whole_area_y=R_moji_y_2.size
 	R_whole_area_m=R_moji_m_2.size
 	R_whole_area_c=R_moji_c_2.size
 	R_white_area_y=cv2.countNonZero(R_moji_y_2)
 	R_white_area_m=cv2.countNonZero(R_moji_m_2)
 	R_white_area_c=cv2.countNonZero(R_moji_c_2)

	L_whole_area_y=L_moji_y_2.size
 	L_whole_area_m=L_moji_m_2.size
 	L_whole_area_c=L_moji_c_2.size
 	L_white_area_y=cv2.countNonZero(L_moji_y_2)
 	L_white_area_m=cv2.countNonZero(L_moji_m_2)
 	L_white_area_c=cv2.countNonZero(L_moji_c_2)


	if R_white_area_y > 1800 and R_white_area_y < 2100:
 		print('R_Yellow_A')
 	if R_white_area_y > 1000 and R_white_area_y < 1500:
 		print('R_Yellow_B')
 	if R_white_area_y > 2250:
 		print('R_Yellow_C')
 
 	if R_white_area_m > 1800 and R_white_area_m < 2100:
 		print('R_Magenta_A')
 	if R_white_area_m > 1000 and R_white_area_m < 1500:
 		print('R_Magenta_B')
 	if R_white_area_m > 2250:
 		print('R_Magenta_C')
 
 	if R_white_area_c > 1800 and R_white_area_c < 2100:
 		print('R_Cyan_A')
 	if R_white_area_c > 1000 and R_white_area_c < 1500:
 		print('R_Cyan_B')
 	if R_white_area_c > 2250:
 		print('R_Cyan_C')

	if L_white_area_y > 1800 and L_white_area_y < 2100:
 		print('L_Yellow_A')
 	if L_white_area_y > 1000 and L_white_area_y < 1500:
 		print('L_Yellow_B')
 	if L_white_area_y > 2250:
 		print('L_Yellow_C')
 
 	if L_white_area_m > 1800 and L_white_area_m < 2100:
 		print('L_Magenta_A')
 	if L_white_area_m > 1000 and L_white_area_m < 1500:
 		print('L_Magenta_B')
 	if L_white_area_m > 2250:
 		print('L_Magenta_C')
 
 	if L_white_area_c > 1800 and L_white_area_c < 2100:
 		print('L_Cyan_A')
 	if L_white_area_c > 1000 and L_white_area_c < 1500:
 		print('L_Cyan_B')
 	if L_white_area_c > 2250:
 		print('L_dCyan_C')
	

        cv2.imshow('R_Y', R_moji_y)
	cv2.imshow('R_C', R_moji_c)
	cv2.imshow('R_M', R_moji_m)

	cv2.imshow('L_Y', L_moji_y)
	cv2.imshow('L_C', L_moji_c)
	cv2.imshow('L_M', L_moji_m)

	#cv2.imshow('Y2', R_moji_y_2)
	#cv2.imshow('C2', R_moji_c_2)
	#cv2.imshow('M2', R_moji_m_2)
	
	#cv2.imshow('Y2', L_moji_y_2)
	#cv2.imshow('C2', L_moji_c_2)
	#cv2.imshow('M2', L_moji_m_2)


        #builder = pyocr.builders.TextBuilder(tesseract_layout=5)
        #result = tool.image_to_string(pil_img, lang="eng", builder=builder)

        #if 'c' in result:
        #    result = 'C'
        #elif 'C' in result:
        #    result = 'C'
        #elif 'E' in result:
        #    result = 'B'
        #else:
        #    result='A'

        #print(result)
        cv2.waitKey(1)
    except Exception as err:
        print err

def start_node():
    rospy.init_node('img_proc_LR')
    rospy.loginfo('img_proc node started')
    rospy.Subscriber("/camera/color/image_raw/compressed", CompressedImage, process_image)
    rospy.spin()

if __name__ == '__main__':
    try:
        start_node()
    except rospy.ROSInterruptException:
        pass
