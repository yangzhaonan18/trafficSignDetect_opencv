# -*- coding=utf-8 -*-
# py3
# 时间：2019年8月7日20:02:02
# 分水岭算法，分割连续相连的红色交通标志

import numpy as np
import cv2


def draw_rectangle(image, rectangleList):
    for i in range(len(rectangleList)):
        x = rectangleList[i][0]
        y = rectangleList[i][1]
        w = rectangleList[i][2]
        h = rectangleList[i][3]
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # print("x, y, w, h ", x, y, w, h)
    cv2.imshow("final result", image)
    cv2.waitKey(0)  # 等待用户操作，里面等待参数是毫秒，我们填写0，代表是永远，等待用户操作
    cv2.destroyAllWindows()  # 销毁所有窗口 按ESC，结束


def find_obj(image, ret, markers):
    # print("ret = ", ret)  # 所有的对象，1表示背景， 2是第一个目标
    rectangleList = []  # 存储 目标的外界矩形信息
    print("ret = ", ret)
    for i in range(2, ret + 1):
        # print("i = ", i)
        black_img = np.zeros_like(image)
        black_img[markers == i] = [255, 255, 255]
        gray = cv2.cvtColor(black_img, cv2.COLOR_BGR2GRAY)  # 转成灰色图像
        ret, BinThings = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)  # 灰色图像二值化（变黑白图像）
        _, contours, hierarchy = cv2.findContours(BinThings, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # cv2.drawContours(image, [contours[0]], -1, (255, 0, 255), 1)  # 话轮廓的边界
        print("len(contours) = ", len(contours))
        if len(contours) > 0:
            x, y, w, h = cv2.boundingRect(contours[0])
            rectangleList.append([x, y, w, h])

    return rectangleList


def watershed_demo(image):
    """
    :param image:  原始RGB 图像
    :return:
    """
    blur = image
    blur = cv2.pyrMeanShiftFiltering(image, 10, 100)  # 滤波
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)  # 获取灰度图像

    ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # 对二值图进行 形态学操作，进一步消除图像中噪点
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mb = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)  # iterations连续两次开操作
    sure_bg = cv2.dilate(mb, kernel, iterations=1)  # 3次膨胀,可以获取到大部分都是背景的区域
    # cv2.imshow("sure_bg", sure_bg)
    # 距离变换
    dist = cv2.distanceTransform(mb, cv2.DIST_L2, 5)
    # cv2.imshow("dist", dist)
    dist_output = cv2.normalize(dist, 0, 1.0, cv2.NORM_MINMAX)  # 这个归一化看不懂？？？？？？
    # print(mb[150][120:140])
    # print(dist[150][120:140])
    # print(dist_output[150][120:140])
    # cv.imshow("distinct-t", dist_output * 50)  # 为什么是50 ？？？？？
    ret, sure_fg = cv2.threshold(dist, dist.max() * 0.6, 255, cv2.THRESH_BINARY)  # 设置阈值进行二值化
    # cv2.imshow("sure_fg", sure_fg)
    # print(sure_fg[150][120:140])
    # print(sure_bg[150][120:140])
    # 获取未知区域
    surface_fg = np.uint8(sure_fg)  # 保持色彩空间一致才能进行运算，现在是背景空间为整型空间，前景为浮点型空间，所以进行转换
    unknown = cv2.subtract(sure_bg, surface_fg)  # 作差
    # cv2.imshow("unkown", unknown)
    # 获取maskers,在markers中含有种子区域
    ret, markers = cv2.connectedComponents(surface_fg)
    # print("ret:", ret)  # 10

    # 分水岭变换
    markers = markers + 1  # 加1 之后，1表示背景。2是第一个目标
    markers[unknown == 255] = 0
    markers = cv2.watershed(image, markers=markers)
    # image[markers == -1] = [0, 0, 255]
    return ret, markers


def find_water_objects(image):
    ret, marskers = watershed_demo(image)  # 使用分水岭算法那找到检测目标的数量和对应的 marskers 标记。
    rectangleList = find_obj(image, ret, marskers)
    # draw_rectangle(image, rectangleList)
    return rectangleList



if __name__ == "__main__":
    image = cv2.imread("image//coin2.png")  # 读取图片
    cv2.namedWindow("input image", cv2.WINDOW_AUTOSIZE)  # 创建GUI窗口,形式为自适应
    cv2.imshow("input image", image)  # 通过名字将图像和窗口联系

    find_water_objects(image)




