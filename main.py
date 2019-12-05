import cv2
import numpy as np
import random as rand
import math

BLUE = (255, 0, 0)
original_img = "test_pvk_wyciagniecie.jpg"
# original_img = "test_srebro.jpg"


# it might be a good idea to rewrite this function?
def find_flask_walls(img, threshold, check_range, repeats=6, margin=0):
    first_wall = []
    second_wall = []
    for l in range(repeats):
        x = rand.randrange(400, img.shape[0] - 400, 1)

        for k in range(img.shape[1] - check_range - margin):
            is_wall = bool(1)
            for j in range(check_range):
                if img[x, margin + k + j] <= threshold:
                    is_wall = bool(0)
                    break
            if is_wall:
                first_wall.append(k + margin)
                break

        for k in range(img.shape[1] - check_range - margin):
            is_wall = bool(1)
            for j in range(check_range):
                if img[x, img.shape[1] - k - j - margin - 1] <= threshold:
                    is_wall = bool(0)
                    break
            if is_wall:
                second_wall.append(img.shape[1] - k - margin - 1)
                break
    return int(np.average(first_wall)), int(np.average(second_wall))


def find_meniscus_top(heights, x1, x2):
    tx = np.argmax(heights[x1:x2]) + x1
    ty = heights[tx]

    return int(tx), int(ty)


def find_meniscus_height(h, tx, ty, margin=0):
    left_meniscus_height = -1
    right_meniscus_height = -1

    # TODO it might be useful to check in what rate the the meniscus grows, to decide whether we want to skip it,
    # otherwise, a constraint that both left and right heights should be similar up to some difference epsilon specified
    # in some way - either by the user as a flag or in some config file or hardcoded in the programme, should be neccessary

    # find lowest value on the left side of the top
    left_meniscus_height = int(abs(h[margin + np.argmin(h[margin:tx])] - ty))
    # do the same with the right hand side height
    right_meniscus_height = int(abs(h[tx + np.argmin(h[tx:-margin])] - ty))

    return left_meniscus_height, right_meniscus_height


def remove_flask_walls(h, threshold_multiplier):
    thresh = np.average(h) * threshold_multiplier
    print("thresh: " + str(thresh))
    middle = int(math.floor(h.shape[0] / 2))

    x = h[middle - 1::-1].shape[0] - np.argmin(h[middle - 1::-1] >= thresh)
    y = middle + np.argmin(h[middle:] >= thresh)
    print("x y: " + str(x) + " " + str(y))

    return h[x:y]


def filter_image(the_img, min_size=15000):
    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(the_img)
    sizes = stats[1:, -1]; nb_components = nlabels - 1

    new_img = np.zeros(the_img.shape)
    for i in range(0, nb_components):
        if sizes[i] >= min_size:
            new_img[labels == i + 1] = 255
    return new_img


def find_edge_height(img):
    new_img = np.fliplr(np.swapaxes(img, 1, 0))
    return np.array(list(map(lambda arr: np.argmax(arr > 180), new_img)))


if __name__ == "__main__":
    img = cv2.imread(original_img, 0)
    cv2.imwrite('_original_image.jpg', img)
    img = cv2.blur(img, (5, 5))

    first_wall, second_wall = find_flask_walls(img, 180, 10, repeats=15, margin=50)
    print(first_wall, second_wall)
    img = img[:, first_wall:second_wall]
    cv2.imwrite('cropped.jpg', img)

    # pre-process and use canny
    img = cv2.blur(img, (15, 15))
    clahe = cv2.createCLAHE(clipLimit=2.00, tileGridSize=(11, 11))
    img = clahe.apply(img)
    ret2, detected_edges = cv2.threshold(img, 10, 230, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bin_img = filter_image(detected_edges)

    cv2.imwrite('bin.jpg', bin_img)
    heights = find_edge_height(bin_img)
    print(heights.shape)

    heights = remove_flask_walls(heights, 0.85)
    print(heights.shape)
    print(heights)
    # dst = cv2.bitwise_and(img, img, mask=edges)
    cv2.imwrite('detected_edges.jpg', detected_edges)

    top_x, top_y = find_meniscus_top(heights, int(heights.shape[0] / 2 - 100), int(heights.shape[0] / 2 + 100))
    print(top_x, top_y)
    left_height, right_height = find_meniscus_height(heights, top_x, top_y, margin=20)
    print(left_height, right_height)
