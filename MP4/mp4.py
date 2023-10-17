import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

Skin_H = {}

def train(path, mode="HSV"):
    img_test = cv.imread(path)
    points = []

    def click_event(event, x, y, flags, params):
        nonlocal points
        if event == cv.EVENT_LBUTTONDOWN:
            points = [(x,y)]
        elif event==cv.EVENT_LBUTTONUP:
            points.append((x,y))
            cv.rectangle(img_test, points[0], points[1], (255, 0, 0), 1)
            cv.imshow("Select data", img_test)

    cv.imshow('Select data', img_test)
    cv.setMouseCallback('Select data', click_event)
    cv.waitKey(0)
    cv.destroyAllWindows()

    if len(points) == 2:
        start_point, end_point = sorted(points, key=lambda p: (p[1], p[0]))
        crop = img_test[start_point[1]+1:end_point[1], start_point[0]+1:end_point[0]]


        if mode == "RGB":
            process_rgb(crop)
        elif mode == "nRGB":
            process_nrgb(crop)
        elif mode == "HSV":
            process_hsv(crop)

def process_rgb(crop):
    crop_height, crop_width, _ = crop.shape
    graph_x, graph_y = [], []
    m = (0, ('R','G'))
    for i in range(crop_height):
        for j in range(crop_width):
            (R,G) = crop[i,j][0:2]
            Skin_H[(R,G)] = Skin_H.get((R,G),0) + 1
            graph_x.append(R)
            graph_y.append(G)
            if Skin_H[(R,G)] > m[0]:
                m = (Skin_H[(R,G)], (R,G))
    create_histogram(graph_x, graph_y, 'RGB')
    normalize_histogram(m)

def process_nrgb(crop):
    crop = crop / 255
    crop_height, crop_width, _ = crop.shape
    graph_x, graph_y = [], []
    m = (0, ('R','G'))
    for i in range(crop_height):
        for j in range(crop_width):
            (R,G) = crop[i,j][0:2]
            Skin_H[(R,G)] = Skin_H.get((R,G),0) + 1
            graph_x.append(R)
            graph_y.append(G)
            if Skin_H[(R,G)] > m[0]:
                m = (Skin_H[(R,G)], (R,G))
    create_histogram(graph_x, graph_y, 'nRGB')
    normalize_histogram(m)

def process_hsv(crop):
    crop = cv.cvtColor(crop, cv.COLOR_BGR2HSV)
    crop_height, crop_width, _ = crop.shape
    graph_x, graph_y = [], []
    m = (0, ('H','S'))
    for i in range(crop_height):
        for j in range(crop_width):
            (H,S) = crop[i,j][0:2]
            Skin_H[(H,S)] = Skin_H.get((H,S),0) + 1
            graph_x.append(H)
            graph_y.append(S)
            if Skin_H[(H,S)] > m[0]:
                m = (Skin_H[(H,S)], (H,S))
    create_histogram(graph_x, graph_y, 'HSV')
    normalize_histogram(m)

def create_histogram(graph_x, graph_y, mode):
    fig = plt.figure()
    plt.hist2d(graph_x, graph_y, bins=(30,30))
    plt.colorbar()
    fig.suptitle(mode)
    fig.savefig(f'hist_{mode}.jpg')

def normalize_histogram(m):
    for key in Skin_H:
        Skin_H[key] = Skin_H[key]/m[0]

def test(path, threshold=0.05, mode='', title=""):
    img_test = cv.imread(path)
    res_height, res_width, _ = img_test.shape
    result_image = np.full((res_height, res_width, 3), 255, dtype=np.uint8)


    if mode == 'RGB':
        rgb_img = img_test.copy()
        for i in range(res_height):
            for j in range(res_width):
                (R,G) = rgb_img[i,j][0:2]
                if (R,G) in Skin_H and Skin_H[(R,G)] > threshold:
                    result_image[i][j] = img_test[i][j]
    elif mode == 'nRGB':
        rgb_img = img_test.copy()/255
        for i in range(res_height):
            for j in range(res_width):
                (R,G) = rgb_img[i,j][0:2]
                if (R,G) in Skin_H and Skin_H[(R,G)] > threshold:
                    result_image[i][j] = img_test[i][j]
    elif mode == 'HSV':
        hsv_img = cv.cvtColor(img_test, cv.COLOR_BGR2HSV)
        for i in range(res_height):
            for j in range(res_width):
                (H,S) = hsv_img[i,j][0:2]
                if (H,S) in Skin_H and Skin_H[(H,S)] > threshold:
                    result_image[i][j] = img_test[i][j]

    cv.imshow("result", result_image)
    cv.imwrite(title + '.jpg', result_image)
    cv.waitKey(0)
    cv.destroyAllWindows()

def main():
    # Training
    train('skin_test3.jpg', mode="RGB")
    train('skin_test3.jpg', mode="nRGB")
    train('skin_test3.jpg', mode="HSV")



    # Testing
    # test('gun1.bmp', threshold=.025, mode="RGB", title="rgb_gun")
    # test('joy1.bmp', mode="RGB", title="rgb_joy")
    # test('pointer1.bmp', mode="RGB", title="rgb_pointer")

    # test('gun1.bmp', threshold=.025, mode="nRGB", title="nrgb_gun")
    # test('joy1.bmp', mode="nRGB", title="nrgb_joy")
    # test('pointer1.bmp', mode="nRGB", title="nrgb_pointer")

    # test('gun1.bmp', threshold=.025, mode="HSV", title="hsv_gun")
    # test('joy1.bmp', mode="HSV", title="hsv_joy")
    # test('pointer1.bmp', mode="HSV", title="hsv_pointer")

if __name__ == "__main__":
    main()

