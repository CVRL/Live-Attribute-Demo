from math import *


def areaOf(coords):
    x1, y1, w1, h1 = coords
    area = w1*h1
    return area


def centerOf(coords):
    x, y, w, h = coords
    center = int(x+(w/2)), int(y+(h/2))
    return center


def angleBetween(coords):
    x1, y1, x2, y2 = coords
    dist_x, dist_y = (x2-x1), (y2-y1)
    angle = (degrees(atan2(dist_y, dist_x)))
    return angle


def midpointOf(coords):
    (x1, x2), (y1, y2) = coords
    mid_point = (int((x1+x1)/2), int((y1+y2)/2))
    return mid_point

