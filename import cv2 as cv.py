import cv2 as cv
import collections

def CCL(img):
    height, width = img.shape
    label = [[0 for _ in range(width)] for _ in range(height)]
    E_table = collections.defaultdict(int)
    L = 0

    # First scanning
    for u in range(height):
        for v in range(width):
            if img[u, v] == 255:  # if pixel is white
                Lu = label[u-1][v] if u-1 >= 0 else 0  # upper label
                Ll = label[u][v-1] if v-1 >= 0 else 0  # left label

                if Lu == Ll and Lu != 0:  # the same label
                    label[u][v] = Lu
                elif Lu != Ll and (Lu == 0 or Ll == 0):  # either is 0
                    label[u][v] = max(Lu, Ll)
                elif Lu != Ll and Lu > 0 and Ll > 0:  # both
                    label[u][v] = min(Lu, Ll)
                    E_table[Lu] = Ll
                else:  # none
                    L += 1
                    label[u][v] = L

    # Second scanning - Renumbering the labels using the E_table
    for u in range(height):
        for v in range(width):
            if label[u][v] in E_table:
                label[u][v] = E_table[label[u][v]]

    return label

def main():
    path = "images/gun.bmp"
    img = cv.imread(path, cv.IMREAD_GRAYSCALE)
    label_img = CCL(img)

    # Convert labels to grayscale for visualization
    max_label = max(map(max, label_img))
    for u in range(len(label_img)):
        for v in range(len(label_img[0])):
            label_img[u][v] = int(label_img[u][v] * 255 / max_label) if max_label != 0 else 0

    title = (path.split('/')[-1])[:-4] + '_labeled.jpg'
    cv.imwrite(title, label_img)
    cv.imshow(title, label_img)
    cv.waitKey(0)
    print('title:', title)

if __name__ == '__main__':
    main()
