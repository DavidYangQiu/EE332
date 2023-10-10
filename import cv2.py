import cv2
import numpy as np

def CCL(img):
    label = 0
    img_label = np.zeros_like(img, dtype=int)
    equivalence = {}

    # First pass
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i, j] == 1:  # Foreground pixel
                neighbors = []
                if i > 0 and img_label[i - 1, j] > 0:
                    neighbors.append(img_label[i - 1, j])
                if j > 0 and img_label[i, j - 1] > 0:
                    neighbors.append(img_label[i, j - 1])

                if not neighbors:
                    label += 1
                    img_label[i, j] = label
                else:
                    min_label = min(neighbors)
                    img_label[i, j] = min_label
                    for l in neighbors:
                        equivalence[l] = min_label

    # Second pass
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img_label[i, j] > 0:
                img_label[i, j] = equivalence[img_label[i, j]]

    num = len(set(equivalence.values()))
    return img_label, num


def main():
    img_names = ['test.bmp', 'face.bmp', 'gun.bmp']
    for name in img_names:
        img = cv2.imread(name, cv2.IMREAD_GRAYSCALE)
        _, binary_img = cv2.threshold(img, 127, 1, cv2.THRESH_BINARY)
        labeled_img, num = CCL(binary_img)
        print(f'{name}: {num} connected components found')
        cv2.imshow(f'Labeled {name}', labeled_img * (255 // num))
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
