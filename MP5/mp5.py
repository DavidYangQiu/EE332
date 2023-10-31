import cv2 as cv
import numpy as np
from scipy import signal
import sys

####PART ONE######
#### GaussSmooth ######
def GaussSmoothing(img, N, S):
    height, width = N
    c = [height // 2, width // 2] #center
    k = np.zeros((height, width))
    for i in range(height):
        for j in range(width):
            x = i - c[0]
            y = j - c[1]
            k[i][j] = np.exp(-(x ** 2 + y ** 2) / (2 * (S ** 2))) / (2 * np.pi * (S ** 2))
    k /= np.sum(k)
    return (signal.convolve2d(img, k)).astype(np.uint8)

####PART two######
#### Calculating Image Gradient ######
def ImageGradient(img, decetor):
    if decetor == 'robert': # [2*2] kernel
        x_kernel = np.array([[1, 0], [0, -1]])
        y_kernel = np.array([[0, -1], [1, 0]])
    elif decetor == 'sobel': #[3*3] kernel
        x_kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        y_kernel = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    convo_img = signal.convolve2d(img, x_kernel + y_kernel)#gradient
    img_mag = np.absolute(convo_img)#magnitude
    img_dirt = np.degrees(np.angle(convo_img))
    return img_mag, img_dirt

####PART three######
#### Selecting High and Low Thresholds ######
def FindThreshold(Mag):
    percentageOfNonEdge = 0.9 #magic number
    histogram = np.zeros(256)
    for val in Mag.flatten():
        histogram[min(int(val), 255)] += 1  # Limit the value to 255
    histogram /= np.sum(histogram)
    cumulative_sum = 0
    for i in range(np.max(Mag)):
        if cumulative_sum >= percentageOfNonEdge:
            T_high = i
            break
        cumulative_sum += histogram[i]
    return T_high, T_high / 2

####PART four######
#### Supressing Nonmaxima ######
def NonmaximaSupress(Mag, angle):
    rows, cols = Mag.shape
    nonm_img = np.zeros((rows, cols), dtype=np.int32)
    angle[angle < 0] += 180

    for x in range(1, rows - 1):
        for y in range(1, cols - 1):
            if (0 <= angle[x, y] < 22.5) or (157.5 <= angle[x, y] <= 180):
                adjacent1 = Mag[x, y-1]
                adjacent2 = Mag[x, y+1]
            elif 22.5 <= angle[x, y] < 67.5:
                adjacent1 = Mag[x-1, y+1]
                adjacent2 = Mag[x+1, y-1]
            elif 67.5 <= angle[x, y] < 112.5:
                adjacent1 = Mag[x-1, y]
                adjacent2 = Mag[x+1, y]
            else:
                adjacent1 = Mag[x-1, y-1]
                adjacent2 = Mag[x+1, y+1]

            if Mag[x, y] >= adjacent1 and Mag[x, y] >= adjacent2:
                nonm_img[x, y] = Mag[x, y]

    return nonm_img

####PART five######
#### Thresholding and Edge Linking ######
def EdgeLinking(suppressed, high_threshold, low_threshold):
    rows, cols = suppressed.shape
    high = (suppressed > high_threshold) * suppressed
    low = (suppressed > low_threshold) * suppressed
    edges = np.copy(high)
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1), (-1, 1), (1, -1)]

    stack = [(x, y) for x in range(rows) for y in range(cols) if low[x, y] and not edges[x, y]]

    while stack:
        x, y = stack.pop()
        edges[x, y] = low[x, y]
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < rows and 0 <= ny < cols and low[nx, ny] and not edges[nx, ny]:
                stack.append((nx, ny))

    return edges.astype(np.uint8)


def main(path):
    img = cv.imread(path, cv.IMREAD_GRAYSCALE)
    
    gau_img = GaussSmoothing(img, [5, 5], 5)
    # cv.imshow('[GaussSmooth]' , gau_img)
    # cv.imwrite('[GaussSmooth] .png', gau_img)
    
    Mag, direction = ImageGradient(gau_img, 'sobel')
    # cv.imshow('[ImageGradient] Magnitude_Sobel ', (Mag/np.max(Mag) * 255).astype(np.uint8))
    # cv.imwrite('[ImageGradient] Magnitude_Sobel .png', (magnitude/np.max(Mag) * 255).astype(np.uint8))
    
    T_high, T_low = FindThreshold(Mag)
    
    suppressed = NonmaximaSupress(Mag, direction)
    # cv.imshow('[NonmaximaSupress]' , suppressed.astype(np.uint8))
    # cv.imwrite('[NonmaximaSupress] .png', suppressed.astype(np.uint8))
    
    edge_link = EdgeLinking(suppressed, T_high, T_low)
    # cv.imshow('[EdgeLinking]', edge_link)
    # cv.imwrite('[EdgeLinking] .png', edge_link)
    cv.waitKey(0)

if __name__ == "__main__":
    main('lena.bmp')
