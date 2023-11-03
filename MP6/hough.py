import numpy as np
import matplotlib.pyplot as plt
import cv2
import scipy.ndimage as ndimage
from scipy.ndimage import filters, label, find_objects

def edge_detect(img,title):
    # Canny Edge Detection  
    edges = cv2.Canny(I_orig, 100, 200)

    #display part 1:edge
    cv2.imshow('canny edge',edges)
    cv2.imwrite('[Edge]'+title+'.png',edges)
    return edges


def para_space(img,title):
    # Hough Transform Initialization
    height, width = img.shape

    theta_res = 360  # Higher theta resolution
    rho_res = 2 * int(np.sqrt(height**2 + width**2))  # Higher rho resolution

    thetas = np.deg2rad(np.linspace(-180, 180, theta_res))
    rhos = np.linspace(-np.sqrt(height**2 + width**2), np.sqrt(height**2 + width**2), rho_res)

    m_c_img = np.zeros((theta_res, rho_res))

    # Populate the Hough Parameter Space
    for x in range(height):
        for y in range(width):
            if img[x, y] == 255:
                for theta_idx in range(theta_res):
                    rho = x * np.cos(thetas[theta_idx]) + y * np.sin(thetas[theta_idx])
                    rho_idx = int(np.round(rho + np.sqrt(height**2 + width**2)))
                    m_c_img[theta_idx, rho_idx] += 1

    #diplay part2: parameter space
    m_c_img_show= (m_c_img/np.max(m_c_img)*255).astype(np.uint8)
    cv2.imshow('parameter space',m_c_img_show)
    cv2.imwrite('[parameter space]'+title+'.png',m_c_img_show)

    return m_c_img, theta_res, thetas, rho_res


def detect_intersections(A, title, threshold):
    filter_size = 2
    data_max = filters.maximum_filter(A, filter_size)
    data_min = filters.minimum_filter(A, filter_size)
    
    maxima = (A == data_max)
    diff = ((data_max - data_min) > threshold)
    maxima[diff == 0] = 0
    
    labeled, num_objects = ndimage.label(maxima)
    slices = ndimage.find_objects(labeled)
    
    intersections = []
    for dy, dx in slices:
        x_center = (dx.start + dx.stop - 1) / 2
        y_center = (dy.start + dy.stop - 1) / 2
        intersections.append((x_center, y_center))

    #diplay part3: significant intersections
    intersections_img = np.zeros(A.shape)
    for x, y in intersections:
        intersections_img[int(y), int(x)] = 255
    cv2.imshow('Intersections', intersections_img)
    cv2.imwrite('[Intersections]'+title+'.png', intersections_img)
    cv2.waitKey(0)
    
    return intersections


def detect_lines(image, m_c_img, title, theta_res, thetas, rho_res):
    height, width = image.shape
    A_copy = np.copy(m_c_img)
    lines_found = 0
    max_lines = 8

    plt.figure()
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(f'{title}')

    while np.max(A_copy) > 0 and lines_found < max_lines:
        max_A = np.max(A_copy)
        peak = np.argwhere(A_copy == max_A)[0]
        t, r = peak
        r = r - np.sqrt(height**2 + width**2)
        if thetas[t] >= np.deg2rad(45) and thetas[t] < np.deg2rad(135):
            x = np.arange(max(height, width))
            y = (r - x * np.cos(thetas[t])) / np.sin(thetas[t])
        else:
            y = np.arange(max(height, width))
            x = (r - y * np.sin(thetas[t])) / np.cos(thetas[t])

        plt.plot(y, x)

        t_range = range(max(0, t-10), min(theta_res, t+11))
        r_range = range(max(0, int(r+np.sqrt(height**2 + width**2))-10), min(rho_res, int(r+np.sqrt(height**2 + width**2))+11))
        A_copy[np.ix_(t_range, r_range)] = 0
        lines_found += 1


    plt.savefig('[Detect Line]'+title+'.png')
    plt.show()


# Load the image
test_img, title = (cv2.imread('test.bmp', cv2.IMREAD_GRAYSCALE), 'test')
test2_img, title2 = (cv2.imread('test2.bmp', cv2.IMREAD_GRAYSCALE), 'test2')
input_img, title3 = (cv2.imread('input.bmp', cv2.IMREAD_GRAYSCALE), 'input')
title4 = 'theta&rho'
I_orig, title = test_img, title4

edges_img = edge_detect(I_orig, title)
m_c_img, theta_res, thetas, rho_res = para_space(edges_img, title)

# Assume m_c_img is your Hough parameter space matrix
intersections = detect_intersections(m_c_img, title, 40)


# Assuming intersections is your list of (t, r) values
detect_lines(I_orig, m_c_img, title, theta_res, thetas, rho_res)
