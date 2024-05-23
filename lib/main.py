import cv2

superpixelLSC = cv2.ximgproc.createSuperpixelLSC(
    I,
    region_size=size,
    ratio=0.075)
