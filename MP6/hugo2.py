# import cv2
# import numpy as np
# from scipy.ndimage import gaussian_filter
# import matplotlib.pyplot as plt

# class Solution():
#     def __init__(self, path, title):
#         self.img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
#         self.title = title

#     def hough(self, theta_res, rho_res, threshold):
#         edges = cv2.Canny(self.img, 60, 120)
#         height, width = self.img.shape
#         max_dist = np.sqrt(height**2 + width**2)
#         theta = np.deg2rad(np.arange(0, 180, theta_res))
#         rho = np.arange(0, max_dist, rho_res)
#         accumulator = np.zeros((len(rho), len(theta)))

#         y_idxs, x_idxs = np.nonzero(edges)  # non-zero edge points
#         for i in range(len(x_idxs)):
#             x = x_idxs[i]
#             y = y_idxs[i]
#             for j in range(len(theta)):
#                 rho_val = x * np.cos(theta[j]) + y * np.sin(theta[j])
#                 rho_idx = np.argmin(np.abs(rho - rho_val))
#                 accumulator[rho_idx, j] += 1

#         # Find local maxima in the accumulator
#         accumulator_smoothed = gaussian_filter(accumulator, sigma=1)
#         local_maxima = (accumulator_smoothed > threshold) & (accumulator_smoothed == accumulator_smoothed.max())

#         rho_idxs, theta_idxs = np.nonzero(local_maxima)
#         found_rho = rho[rho_idxs]
#         found_theta = theta[theta_idxs]

#         # Plotting
#         plt.figure(figsize=(10, 10))

#         # Plot Hough Transform
#         plt.subplot(121)
#         plt.imshow(accumulator, extent=[np.rad2deg(theta[-1]), np.rad2deg(theta[0]), rho[-1], rho[0]], cmap='hot', aspect='auto')
#         plt.title('Hough Transform')
#         plt.xlabel('Theta (degrees)')
#         plt.ylabel('Rho (pixels)')
#         plt.colorbar()

#         # Plot Detected Lines
#         plt.subplot(122)
#         plt.imshow(edges, cmap='gray')
#         plt.title('Detected Lines')
#         for i in range(len(found_rho)):
#             a = np.cos(found_theta[i])
#             b = np.sin(found_theta[i])
#             x0 = a * found_rho[i]
#             y0 = b * found_rho[i]
#             x1 = int(x0 + 1000 * (-b))
#             y1 = int(y0 + 1000 * (a))
#             x2 = int(x0 - 1000 * (-b))
#             y2 = int(y0 - 1000 * (a))
#             plt.plot((x1, x2), (y1, y2), '-r')

#         plt.xlim([0, width])
#         plt.ylim([height, 0])
#         plt.show()

# # Usage
# soln = Solution('input.bmp', 'input')
# soln.hough(1, 1, 50)
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
I_orig = cv2.imread('input.bmp', cv2.IMREAD_GRAYSCALE)

# Display the original image
plt.figure()
plt.imshow(I_orig, cmap='gray')
plt.title('Original Image')

# Edge Detection
edges = cv2.Canny(I_orig, 100, 200)

# Hough Transform Initialization
M, N = edges.shape
r_min = np.round(-np.sqrt(M**2 + N**2))
r_min = np.abs(r_min) + 1
r_max = r_min + np.round(np.sqrt(M**2 + N**2))

A = np.zeros((180+181, int(r_max)))

# Populate the Hough Parameter Space
for x in range(M):
    for y in range(N):
        if edges[x, y] == 255:
            for t in range(180):
                r = x * np.cos(np.deg2rad(t)) + y * np.sin(np.deg2rad(t))
                r = np.round(r)
                A[t, int(r+r_min)] += 1

# Gaussian Smoothing
A_smooth = cv2.GaussianBlur(A, (5, 5), 0.6)

# def detect_lines(A, title):
#     # Find the 1st peak
#     max_A = np.max(A)
#     peaks = np.argwhere(A == max_A)
#     for peak in peaks:
#         t, r = peak
#         r = r - r_min
#         if t >= 45 and t < 135:
#             x = np.arange(max(M, N))
#             y = np.round((r - x * np.cos(np.deg2rad(t))) / np.sin(np.deg2rad(t)))
#         else:
#             y = np.arange(max(M, N))
#             x = np.round((r - y * np.sin(np.deg2rad(t))) / np.cos(np.deg2rad(t)))

#         # Plot the detected lines
#         plt.figure()
#         plt.imshow(edges, cmap='gray')
#         plt.plot(y, x, 'r')
#         plt.title(title)

#         # Update the output image
#         out = np.ones((M, N))
#         for p in range(len(x)):
#             if 0 <= x[p] < M and 0 <= y[p] < N:
#                 out[int(x[p]), int(y[p])] = 0

#         # Display the result
#         plt.figure()
#         plt.imshow(out, cmap='gray')
#         plt.show()
import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.ndimage import convolve, gaussian_filter
from scipy.signal import argrelextrema

def detect_lines(image,A, title):
    # Copy A to avoid modifying the original
    A_copy = np.copy(A)
    # Continue until no more peaks are found or a certain number of lines have been found
    lines_found = 0
    max_lines = 4∑  # set a limit to the number of lines to find

    # Create a figure outside the loop
    plt.figure()
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # Assume image is in BGR format
    plt.title(f'{title}')

    while np.max(A_copy) > 0 and lines_found < max_lines:
        # Find the peak
        max_A = np.max(A_copy)
        peak = np.argwhere(A_copy == max_A)[0]
        t, r = peak
        r = r - r_min
        if t >= 45 and t < 135:
            x = np.arange(max(M, N))
            y = np.round((r - x * np.cos(np.deg2rad(t))) / np.sin(np.deg2rad(t)))
        else:
            y = np.arange(max(M, N))
            x = np.round((r - y * np.sin(np.deg2rad(t))) / np.cos(np.deg2rad(t)))

        # Plot the detected lines on the same figure
        plt.plot(y, x, label=f'Line {lines_found + 1}')

        # Remove the detected line from A_copy
        t_range = range(max(0, t-10), min(360, t+11))
        r_range = range(max(0, int(r+r_min)-10), min(int(r_max), int(r+r_min)+11))
        A_copy[np.ix_(t_range, r_range)] = 0
        lines_found += 1

    # Add legend to the plot
    plt.show()
    plt.imsave()
    
# Assume edges, M, N, r_min, r_max are defined
# Call the function
detect_lines(I_orig,A, 'Smoothed Aqweqweqweq')


