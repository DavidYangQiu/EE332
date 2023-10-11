import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def read_image(path):
    return cv.imread(path, cv.IMREAD_GRAYSCALE)

def HistoEqualization(img):
    img_height, img_width = img.shape
    H = [0 for _ in range(256)]

    for i in range(img_height):
        for j in range(img_width):
            H[img[i][j]] += 1

    Histogram = plt.figure()
    plt.hist(img.flatten(), [i for i in range(256)])
    Histogram.suptitle('Histogram')
    Histogram.savefig('Histogram-chart.jpg')

    T = [0 for _ in range(256)]
    T[0] = H[0]

    for i in range(1,256):
        T[i] = T[i-1] + H[i]

    for i in range(256):
        T[i] = T[i]/T[-1]

    Transformation = plt.figure()
    plt.plot([i for i in range(256)], T)
    Transformation.suptitle('Transformation')
    Transformation.savefig('Transformation-curve.jpg')

    new_img = img.copy()
    for i in range(img_height):
        for j in range(img_width):
            color = img[i][j]
            new_img[i][j] = int(T[color] * 255)

    return new_img

def Linear_Lighting(img):
    img_height, img_width = img.shape
    y = img.ravel()
    A = np.column_stack((np.repeat(np.arange(img_height), img_width), np.tile(np.arange(img_width), img_height), np.ones(img_height * img_width)))
    x = np.linalg.lstsq(A, y, rcond=None)[0]

    plane = np.dot(A, x).reshape(img_height, img_width)
    plane = np.clip(plane, 0, 255).astype(np.uint8)

    scaling_factor = 0.5
    diff = (img - plane) * scaling_factor
    corrected = np.clip(img + diff, 0, 255).astype(np.uint8)

    return corrected

def Quadratic_Lighting(img):
    img_height, img_width = img.shape
    i_coords, j_coords = np.repeat(np.arange(img_height), img_width), np.tile(np.arange(img_width), img_height)
    A = np.column_stack((i_coords**2, i_coords * j_coords, j_coords**2, i_coords, j_coords, np.ones(img_height * img_width)))
    y = img.ravel()
    x = np.linalg.lstsq(A, y, rcond=None)[0]

    plane = np.dot(A, x).reshape(img_height, img_width)
    plane = np.clip(plane, 0, 255).astype(np.uint8)

    scaling_factor = 0.5
    diff = (img - plane) * scaling_factor
    corrected = np.clip(img + diff, 0, 255).astype(np.uint8)

    return corrected


def main():
    img = read_image("moon.bmp")
    recolor = HistoEqualization(img)
    cv.imshow('Non-lighting', recolor)
    cv.imwrite('Non-lighting.jpg', recolor)

    # linear_light = Linear_Lighting(img)
    # cv.imshow('Linear Lighting3', linear_light)
    # cv.imwrite('linear-lighting3.jpg', linear_light)

    # quadratic_light = Quadratic_Lighting(recolor)
    # cv.imshow('Quadratic Lighting3', quadratic_light)
    # cv.imwrite('quadratic-lighting3.jpg', quadratic_light)
    cv.waitKey(0)

if __name__ == "__main__":
    main()





