import fnmatch
from PIL import Image
from PIL import ImageDraw
import numpy as np
from config import *


# area calculation of the polygon
# draw binary map and count pixels
def polygonArea(poly_coor, im_height, im_width):
    img1 = Image.new('1', (im_width, im_height))
    ImageDraw.Draw(img1).polygon(poly_coor, outline=1, fill=1)
    area = np.count_nonzero(np.array(img1))
    return area


# create masks of the truth polygon and our rect and find the area of intersection
# gets
def find_intersection(rect, poly_coor, im_height, im_width):
    # create to images of size im_height X im_width
    img1 = Image.new('1', (im_width, im_height))
    img2 = Image.new('1', (im_width, im_height))

    # make binary map of the poly and bounding box (rect)
    ImageDraw.Draw(img1).polygon(poly_coor, outline=1, fill=1)
    ImageDraw.Draw(img2).polygon(rect, outline=1, fill=1)

    # save option to check code
    # img1.save("C:\\Users\\eyal\\Downloads\\img1.jpg")
    # img1.save("C:\\Users\\eyal\\Downloads\\img2.jpg")

    # calculate the area
    # and of the two arrays to get the intersection
    # then count the left over 1's
    area = np.count_nonzero(np.logical_and(np.array(img1), np.array(img2)))
    return area


# turn the box locations file to list
def get_bounding_box(d_path):
    data_file = os.path.join(d_path, "groundtruth.txt")

    with open(data_file, 'r') as f:
        # list of 4 x,y locations for object
        tmp_pixels = f.readlines()

    # now list of lists
    # in each element there are 4 elements
    # [x1,y1,x2,y2,x3,y3,x4,y4]
    list_pix_loc = [i.split(',') for i in tmp_pixels]
    # all elements in the list are now float
    pix_loc = [list(map(float, i)) for i in list_pix_loc]
    return pix_loc


def clean_folder(d_path, cur_dir):
    for i in os.listdir(os.path.join(d_path, cur_dir)):
        if fnmatch.fnmatch(i, '*_crop*'):
            os.remove(os.path.join(d_path, cur_dir, i))


# for each pic make N crops
def make_crops(d_path, cur_dir, num_crops):
    # create a list of all the pics
    pic_list = [i.strip(".jpg") for i in os.listdir(os.path.join(d_path, cur_dir)) if i.endswith(".jpg")]
    # bounding box location list
    b_box_loc = get_bounding_box(os.path.join(d_path, cur_dir))
    for i in range(len(pic_list)):

        width = int(round(max(*b_box_loc[i][:7:2]) - round(min(*b_box_loc[i][:7:2]))))

        height = int(round(max(*b_box_loc[i][1:8:2])) - round(min(*b_box_loc[i][1:8:2])))
        print(i)
        lft = int(round(min(*b_box_loc[i][:7:2])))
        up = int(round(min(*b_box_loc[i][1:8:2])))
        down = up + height
        right = lft + width
        cur_im = Image.open(os.path.join(d_path, cur_dir, pic_list[i]) + ".jpg")
        im_height = cur_im.height
        im_width = cur_im.width
        true_crop = cur_im.crop((lft, up, right, down))
        true_crop.save(os.path.join(d_path, cur_dir, pic_list[i] + "_true_crop.jpg"))
        for j in range(num_crops):
            new_lft = int(lft) + round((width / 2.5) * np.random.normal(0, 3))
            new_up = int(up) + round((height / 2.5) * np.random.normal(0, 3))
            # find the intersection between our rect and the orig poly of the bounding box
            # find the overlap percentage
            rect = [new_lft, new_up, new_lft+width, new_up, new_lft+width, new_up+height, new_lft, new_up+height]
            overlap = 100 * find_intersection(rect, b_box_loc[i], im_height, im_width) / \
                      polygonArea(b_box_loc[i], im_height, im_width)

            a = ((new_lft < 0) | (new_lft + width > im_width) | (new_up < 0) | (new_up + height > im_height))
            cnt = 0
            while a:
                cnt += 1
                new_lft = int(lft) + round((width / 2.5) * np.random.normal(0, 3))
                new_up = int(up) + round((height / 2.5) * np.random.normal(0, 3))
                rect = [new_lft, new_up, new_lft + width, new_up, new_lft + width, new_up + height, new_lft,
                        new_up + height]
                overlap = 100 * find_intersection(rect, b_box_loc[i], im_height, im_width) / \
                          polygonArea(b_box_loc[i], im_height, im_width)
                a = ((new_lft < 0) | (new_lft + width > im_width) | (new_up < 0) | (new_up + height > im_height))
                if cnt > 10**2:
                    break

            cur_crop = cur_im.crop((new_lft, new_up, new_lft + width, new_up + height))
            im_naming_args = (os.path.join(d_path, cur_dir, pic_list[i]), j, overlap)
            cur_crop.save("%s_crop_%02d_%02d.jpg" % im_naming_args)
            im_args = (pic_list[i], j, new_lft, new_up, (new_lft+width), (new_up+height), (new_up+height))

            with open(os.path.join(d_path, cur_dir)+"\\list_of_crops.txt", 'a') as f:
                f.write("%s_crop_%02d, left=%04d, up=%04d, right=%04d, height=%04d, overlap=%02d\n" % im_args)


# Define the base folder
# project_dir = config.base_dir
project_dir = r'C:\Users\eyal\Desktop\Project_A_new_data\vot_full'
num_crops = 10  # Arbitrarily set to 10

# get the names of all the folder to iterate over
dir_list = [i for i in os.listdir(project_dir) if os.path.isdir(os.path.join(project_dir, i))]

for dirs in dir_list:
    print(dirs)
    # Optional removal of all created pics
    clean_folder(project_dir, dirs)
    # go into each folder and create crops of images with different overlap
    make_crops(project_dir, dirs, num_crops)
