# -*- coding=utf-8 -*-
# py3
# 显示最后的矩形框（候选区域）
import cv2
import os


def drawSign(img, rectangleList):
    for i in range(len(rectangleList)):
        x = rectangleList[i][0]
        y = rectangleList[i][1]
        w = rectangleList[i][2]
        h = rectangleList[i][3]
        crop_img = img[y:y + h, x:x + w]
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # 对裁剪出来的区域进行分类（判断是否是所需的检测对象）
    cv2.imshow("img  with rectangle", img)
    cv2.waitKey(0)


def cutSign(img, List, dir, M):
    for i in range(len(List)):
        x1 = List[i][0]
        y1 = List[i][1]
        x2 = List[i][0] + List[i][2]
        y2 = List[i][1] + List[i][3]
        try:
            Sign_image = img[y1:y2, x1:x2]
            Sign_image = cv2.resize(Sign_image, (28, 28))
            print("%6d" % i)
            path = os.path.join(dir, "%d_%d.jpg" % (M,  i))
            print(path)
            cv2.imwrite(path, Sign_image)
        except:
            pass
