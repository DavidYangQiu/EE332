import cv2 as cv
import numpy as np


def create_SE(dimensions):
#   return cv.getStructuringElement(cv.MORPH_RECT, dimensions)
    return cv.getStructuringElement(cv.MORPH_CROSS, dimensions)


def boundary(img, SE):
    return img - cv.erode(img, SE)

def main():
    img_path = 'images/gun.bmp'
    img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)

    SE = create_SE((5, 5))

    dilation_img = cv.dilate(img, SE)
    cv.imshow('Dilation [3x3]', dilation_img)
    cv.imwrite('dilation [3x3].bmp', dilation_img)
    cv.waitKey(0)

    erosion_img = cv.erode(img, SE)
    cv.imshow('Erosion [3x3]', erosion_img)
    cv.imwrite('erosion [3x3].bmp', erosion_img)
    cv.waitKey(0)

    opening_img = cv.morphologyEx(img, cv.MORPH_OPEN, SE)
    cv.imshow('Opening [3x3]', opening_img)
    cv.imwrite('opening [3x3].bmp', opening_img)
    cv.waitKey(0)

    closing_img = cv.morphologyEx(img, cv.MORPH_CLOSE, SE)
    cv.imshow('Closing [3x3]', closing_img)
    cv.imwrite('closing [3x3].bmp', closing_img)
    cv.waitKey(0)

    boundary_img = img - cv.erode(img, SE)
    cv.imshow('Boundary [3x3]', boundary_img)
    cv.imwrite('boundary [3x3].bmp', boundary_img)
    cv.waitKey(0)

    clean_boundary_gun = boundary(dilation_img, SE)
    cv.imshow('clean_boundary gun', clean_boundary_gun)
    cv.imwrite('results/clean_boundary gun.bmp', clean_boundary_gun)
    cv.waitKey(0)


if __name__ == '__main__':
    main()