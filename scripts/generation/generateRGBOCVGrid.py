import cv2
import numpy as np

dirname = "./rgbocv_grid_outputs/"

black_image = np.zeros((800, 1280, 3), dtype=np.uint8)

image = np.zeros((800, 1280, 3), dtype=np.uint8)
top_left = (490, 250)
bottom_right = (790, 550)
cv2.rectangle(image, top_left, bottom_right, (0, 0, 255), -1)
cv2.imwrite(f'{dirname}/red_RGB.png', image)
cv2.imwrite(f'{dirname}/red_OCV.png', black_image)

cv2.imwrite(f'{dirname}/orange_RGB.png', black_image)
cv2.imwrite(f'{dirname}/orange_OCV.png', image)

# Create an image with a green square in the middle
image = np.zeros((800, 1280, 3), dtype=np.uint8)
cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), -1)

cv2.imwrite(f'{dirname}/green_RGB.png', image)
cv2.imwrite(f'{dirname}/green_OCV.png', black_image)

cv2.imwrite(f'{dirname}/cyan_RGB.png', black_image)
cv2.imwrite(f'{dirname}/cyan_OCV.png', image)


# Create an image with a blue square in the middle
image = np.zeros((800, 1280, 3), dtype=np.uint8)
cv2.rectangle(image, top_left, bottom_right, (255, 0, 0), -1)

cv2.imwrite(f'{dirname}/blue_RGB.png', image)
cv2.imwrite(f'{dirname}/blue_OCV.png', black_image)

cv2.imwrite(f'{dirname}/violet_RGB.png', black_image)
cv2.imwrite(f'{dirname}/violet_OCV.png', image)


# Create a 2x3 grid of equally spaced square
grid_image_RGB = np.zeros((800, 1280, 3), dtype=np.uint8)
grid_image_OCV = np.zeros((800, 1280, 3), dtype=np.uint8)
rows, cols = 2, 3
square_size = min(grid_image_RGB.shape[0] // (rows + 1), grid_image_RGB.shape[1] // (cols + 1))
spacing_y = (grid_image_RGB.shape[0] - rows * square_size) // (rows + 1)
spacing_x = (grid_image_RGB.shape[1] - cols * square_size) // (cols + 1)
colors = [[(0, 0, 255), (0, 0, 0)], [(0, 255, 0), (0, 0, 0)], [
    (255, 0, 0), (0, 0, 0)], [(0, 0, 0), (0, 0, 255)],  [(0, 0, 0), (0, 255, 0)],  [(0, 0, 0), (255, 0, 0)]]

for i in range(rows):
    for j in range(cols):
        top_left = (j * (square_size + spacing_x) + spacing_x, i * (square_size + spacing_y) + spacing_y)
        bottom_right = (top_left[0] + square_size, top_left[1] + square_size)
        color = colors[i * cols + j]
        cv2.rectangle(grid_image_RGB, top_left, bottom_right, color[0], -1)
        cv2.rectangle(grid_image_OCV, top_left, bottom_right, color[1], -1)

cv2.imwrite(f'{dirname}/grid_RGB.png', grid_image_RGB)
cv2.imwrite(f'{dirname}/grid_OCV.png', grid_image_OCV)
