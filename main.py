# -*- coding=utf-8 -*-
# 时间：2019年8月7日20:01:11
# py37
import os
import cv2
from getRectangleList import getRectangleList
from showAndCutSign import drawSign, cutSign

# 读取图片
# image_dir = "D:\\DataSetsH\\2018changshu\\duringGame\\Competition-signal\\TSD-Signal"
image_dir = "image"
save_dir = "outImage"
# img_path = "image//414.png"


names = os.listdir(image_dir)
for j in range(len(names)):
    name = names[j]
    image_path = os.path.join(image_dir,  name)
    print(image_path)
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    # cv2.imshow("asdf", img)
    # cv2.waitKey(0)
    img_draw = cv2.imread(image_path, cv2.IMREAD_COLOR)
    rectangleList = getRectangleList(img)

    # 绘制最后的图像
    drawSign(img_draw, rectangleList)
    # 裁剪出sign
    cutSign(img, rectangleList, save_dir,  j)


print("THE END")

