
import cv2 as cv
import collections
import numpy as np

def connected_component_labeling(binary_img):
    height, width = binary_img.shape
    next_label = 1
    labels = np.zeros((height, width), dtype=int)
    equivalences = collections.defaultdict(int)

    # First pass
    for i in range(height):
        for j in range(width):
            if binary_img[i, j] > 0:
                top = labels[i-1][j] if i > 0 else 0
                left = labels[i][j-1] if j > 0 else 0

                if top and left:
                    if top == left:
                        labels[i][j] = top
                    else:
                        labels[i][j] = min(top, left)
                        equivalences[max(top, left)] = min(top, left)
                else:
                    labels[i][j] = top + left if top or left else next_label
                    if labels[i][j] == next_label:
                        next_label += 1

    # Resolve equivalences
    for key, value in list(equivalences.items()):
        while value in equivalences:
            value = equivalences[value]
        equivalences[key] = value

    # Second pass
    for i in range(height):
        for j in range(width):
            if labels[i][j] in equivalences:
                labels[i][j] = equivalences[labels[i][j]]

    # Calculate areas
    areas = collections.Counter(labels.flatten())

    # Filter by size
    min_area = 400
    for label, area in areas.items():
        if area < min_area:
            labels[labels == label] = 0

    # Create output image
    unique_labels = np.unique(labels)
    max_label = len(unique_labels)
    output_img = np.zeros_like(binary_img)
    for i in range(height):
        for j in range(width):
            if labels[i][j]:
                output_img[i][j] = int(255 * labels[i][j] / max_label)

    return output_img

def main():
    img_path = "gun.bmp"
    img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
    labeled_img = connected_component_labeling(img)
    cv.imwrite('gun_result_filter.jpg', labeled_img)

if __name__ == '__main__':
    main()
