from PIL import Image
from PIL import ImageDraw
import numpy as np


def find_intersection(rect, rect_height, rect_width, poly_coor, poly_height, poly_width):
    # Image.new gets (width,height) tuple
    img1 = Image.new('1', (poly_width, poly_height))
    img2 = Image.new('1', (poly_width, poly_height))
    # img2 = Image.new('1', (rect_width, rect_height))
    ImageDraw.Draw(img1).polygon(poly_coor, outline=1, fill=1)
    ImageDraw.Draw(img2).polygon(rect, outline=1, fill=1)
    img1.save("C:\\Users\\eyal\\Downloads\\img1.jpg")
    img2.save("C:\\Users\\eyal\\Downloads\\img2.jpg")
    area = np.count_nonzero(np.logical_and(np.array(img1), np.array(img2)))
    return area


poly = [0, 0, 15, 0, 15, 15, 0, 21]
poly2 = [4, 0, 13, 0, 12, 15, 0, 21]
left = min(*poly[:7:2])
right = max(*poly[:7:2])
down = max(*poly[1:8:2])
up = min(*poly[1:8:2])
rect = [left, up, right, up, right, down, left, down]

print(find_intersection(poly2, right-left, down-up, poly, max(*poly[1:8:2])-min(*poly[1:8:2]), max(*poly[:7:2])-min(*poly[:7:2])))