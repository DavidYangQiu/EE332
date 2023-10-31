import cv2 as cv
import numpy as np
from scipy import signal
import sys

class Solution():
    def __init__(self, path, title):
        # image
        self.img = cv.imread(path, cv.IMREAD_GRAYSCALE)
        self.title = title
    
####PART ONE######
#### GaussSmooth ######
    def GaussSmoothing(self, N, Sigma):
        # Kernel 
        height = N[0]
        width = N[1]
        #[height,width] value should be odd [1,3,5,...]
        center = [height//2, width//2]

        kernel = np.zeros((height,width))

        # fill kernel
        for i in range(height):
            for j in range(width):
                x = i-center[0]
                y = j-center[0]
                kernel[i][j] = np.exp(-(x**2+y**2)/(2*(Sigma**2)))/(2*np.pi*(Sigma**2))

        # normalize, value sum up to 1
        total = np.sum(kernel)
        kernel = kernel/total
        # image convolution
        smooth_img = (signal.convolve2d(self.img, kernel)).astype(np.uint8)
        # display
        cv.imshow('[GaussSmooth] ' + self.title, np.absolute(smooth_img))
        cv.imwrite('[GaussSmooth] ' + self.title + '.png', np.absolute(smooth_img))
        cv.waitKey(0)

        return smooth_img
    
####PART two######
#### Calculating Image Gradient ######
    def ImageGradient(self, img, mode):
        if mode == 'Robert': # [2*2] kernel
            x_kernel = np.array([[1,0],
                                [0,-1]])
            y_kernel = np.array([[0,-1j],
                                [1j,0]])
            kernel = x_kernel + y_kernel
        elif mode == 'Sobel': #[3*3] kernel
            x_kernel = np.array([[-1,0,1],
                                [-2,0,2],
                                [-1,0,1]])
            y_kernel = np.array([[1j,2j,1j],
                                [0,0,0],
                                [-1j,-2j,-1j]])
            kernel = x_kernel + y_kernel


        # create magnitude and direction images
        magnitude = np.zeros(img.shape)
        direction = np.zeros(img.shape)

        # convolute kernel with image -> note this assumes x_kernel and y_kernel are the same shape
        gradient = signal.convolve2d(img, kernel, )

        # magnitude
        magnitude = np.absolute(gradient)

        # direction in degrees
        direction = np.degrees(np.angle(gradient))

        # Display
        magnitude_img = magnitude/np.max(magnitude) * 255
        magnitude_img = magnitude_img.astype(np.uint8)


        cv.imshow('[ImageGradient] Magnitude_' + mode + ' ' + self.title, magnitude_img)
        cv.imwrite('[ImageGradient] Magnitude_' + mode + ' ' + self.title + '.png', magnitude_img)
        cv.waitKey(0)


        return magnitude, direction
    
####PART three######
#### Selecting High and Low Thresholds ######
    def FindThreshold(self, magnitude, percentageOfNonEdge=0.8):
        self.pne = str(percentageOfNonEdge)
        T_high = 0
        magnitude = magnitude.astype(np.uint8)
        height, width = magnitude.shape
        histogram = np.zeros(256)
        max_n = 0
        h_sum = 0

        # max magnitude
        max_n = np.max(magnitude)

        # build histogram
        for i in range(height):
            for j in range(width):
                histogram[magnitude[i][j]] += 1

        # normalize
        histogram = histogram/np.sum(histogram)

        # find threshold
        for i in range(max_n):
            if h_sum >= percentageOfNonEdge:
                T_high = i
                break
            h_sum += histogram[i]

        #display
        # print("T_high=",T_high)
        return T_high, T_high/2

####PART four######
#### Supressing Nonmaxima ######
    def NonmaximaSupress(self, magnitude, direction):
        height, width = magnitude.shape
        supressed = np.zeros(magnitude.shape)
        direction[direction < 0] += 180


        for i in range(1,height-1):
            for j in range(1,width-1):
                # Horizontal direction
                # Note>L10_edge,page18> 5,1
                if (0 <= direction[i][j] < 22.5) or (157.5 <= direction[i][j] <= 180):
                    p = [0, 1]
                # Diagonal (from top-left to bottom-right): 6,2
                elif (22.5 <= direction[i][j] < 67.5):
                    p = [-1,1]
                # Vertical, 3,7
                elif (67.5 <= direction[i][j] < 112.5):
                    p = [-1,0]
                # Diagonal (from top-right to bottom-left), 4,8
                elif (112.5 <= direction[i][j] < 157.5):
                    p = [-1,-1]

                # if local max -> fill in with magnitude
                if (magnitude[i][j] >= magnitude[i+p[0]][j+p[1]]) and (magnitude[i][j] >= magnitude[i-p[0]][j-p[1]]):
                    supressed[i,j] = magnitude[i][j]

        #Display
        supressed_img = supressed.astype(np.uint8)

        cv.imshow('[NonmaximaSupress] ' + self.title, supressed_img)
        cv.imwrite('[NonmaximaSupress] ' + self.title + '.png', supressed_img)
        cv.waitKey(0)

        return supressed

####PART five######
#### Thresholding and Edge Linking ######
    def EdgeLinking(self, supressed, T_high, T_low):
        # generate mag_high and mag_low
        mag_high = np.zeros(supressed.shape)
        mag_low = np.zeros(supressed.shape)
        height, width = supressed.shape

        for i in range(height):
            for j in range(width):
                if supressed[i][j] > T_high:
                    mag_high[i][j] = supressed[i][j]
                if supressed[i][j] > T_low:
                    mag_low[i][j] = supressed[i][j]

        #Display
        mh = mag_high.astype(np.uint8)
        ml = mag_low.astype(np.uint8)

        cv.imshow('[Threshold High] ' + self.title, mh)
        cv.imwrite('[Threshold High2] ' + self.title + '.png', mh)

        cv.imshow('[Threshold low] ' + self.title, ml)
        cv.imwrite('[Threshold low2] ' + self.title + '.png', ml)
        cv.waitKey(0)

        # recursively generate edge_link
        edge_link = np.copy(mag_high)
        directions = [(1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,-1),(-1,1),(1,-1)]

        # recurse through mag_low, if it has a mag_high link at an adjacent pixel -> add it to edge_link
        sys.setrecursionlimit(10**5)

        def helper(i, j):
            for direction in directions:
                y = i + direction[0]
                x = j + direction[1]
                if y in range(height) and x in range(width) and \
                mag_low[y][x] > 0 and edge_link[y][x] == 0:
                    edge_link[i][j] = mag_low[i][j]
                    helper(y,x)

        for i in range(height):
            for j in range(width):
                if edge_link[i][j] == 0 and mag_low[i][j] > 0:
                    helper(i, j)

        edge_link = edge_link.astype(np.uint8)
        cv.imshow('[EdgeLinking] ' + self.title, edge_link)
        cv.imwrite('[EdgeLinking] ' + self.title + '.png', edge_link)
        cv.waitKey(0)

        print('sanity check', np.array_equal(edge_link, mag_low), np.array_equal(edge_link, mag_high))

        return edge_link

#Comparsion#
# # (N, Sigma)
# test1 = Solution('test1.bmp', 'test1')
# # N_1 = test1.GaussSmoothing([1,1], 1)
# # N_3 = test1.GaussSmoothing([3,3], 1)
# # N_5 = test1.GaussSmoothing([5,5], 1)

# # S_1 = test1.GaussSmoothing([5,5], 1)
# # S_3 = test1.GaussSmoothing([5,5], 3)
# S_5 = test1.GaussSmoothing([5,5], 5)

# # decetor model [robert, sobel]
# smooth_img = S_5
# # magnitude, direction = test1.ImageGradient(smooth_img, mode='Robert')
# magnitude, direction = test1.ImageGradient(smooth_img, mode='Sobel')

# # percentageOfNonEdge value
# T_high, T_low = test1.FindThreshold(magnitude)
# supressed = test1.NonmaximaSupress(magnitude, direction)
# edge_link = test1.EdgeLinking(supressed, T_high, T_low)



lena = Solution('lena.bmp', 'lena')
test1 = Solution('test1.bmp', 'test1')
joy1 = Solution('joy1.bmp', 'joy1')
pointer1 = Solution('pointer1.bmp', 'pointer1')
# select the image to imply
result_image = lena
smooth_img = result_image.GaussSmoothing([3,3], 1)
magnitude, direction = result_image.ImageGradient(smooth_img, mode='Sobel')
T_high, T_low = result_image.FindThreshold(magnitude)
supressed = result_image.NonmaximaSupress(magnitude, direction)
edge_link = result_image.EdgeLinking(supressed, T_high, T_low)

