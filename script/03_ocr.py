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

p1 = np.array([477, 399])
p2 = np.array([663, 399])
p3 = np.array([432, 534])
p4 = np.array([657, 540])

src = np.float32([p1, p2, p3, p4])
dst = np.float32([[0,0],[100,0],[0,100],[100,100]])

M = cv2.getPerspectiveTransform(src, dst)

def process_image(msg):
    try:
        bridge = CvBridge()
        orig = bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
        img = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
        #cv2.imshow('image', img)
        moji = cv2.warpPerspective(img, M,(100,100))
        cv2.imshow('image2', moji)
        pil_img = Image.fromarray(moji)
        builder = pyocr.builders.TextBuilder(tesseract_layout=5)
        result = tool.image_to_string(pil_img, lang="eng", builder=builder)
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