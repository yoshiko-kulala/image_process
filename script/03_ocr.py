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

p1 = np.array([485, 405])
p2 = np.array([655, 405])
p3 = np.array([440, 525])
p4 = np.array([650, 535])

src = np.float32([p1, p2, p3, p4])
dst = np.float32([[0,0],[100,0],[0,100],[100,100]])

M = cv2.getPerspectiveTransform(src, dst)
kernel = np.ones((3, 3), np.uint8)

def process_image(msg):
    try:
        bridge = CvBridge()
        orig = bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
        img = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
        ret, img = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)
        img = cv2.erode(img,kernel,iterations = 3)
        #cv2.imshow('image', img)
        moji = cv2.warpPerspective(img, M,(100,100))
        moji90 = cv2.rotate(moji, cv2.ROTATE_90_CLOCKWISE)
        #moji90 =moji
        pil_img = Image.fromarray(moji90)
        cv2.imshow('image2', moji)
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