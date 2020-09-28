# author: kcgarikipati@gmail.com

"""contains image processing utilities"""

import cv2
import os
import glob
import numpy as np
import pdb


def resize(src_dir, dest_dir, height, width, src_fmt='jpg', dest_fmt='png', interp=cv2.INTER_AREA):
    """
        Resize src image to destination
    """
    if not os.path.exists(dest_dir):
        os.mkdir(dest_dir)

    for filepath in glob.glob(src_dir + '/*.{}'.format(src_fmt)):
        filename = os.path.basename(filepath)
        img = cv2.imread(filepath)
        img_resized = cv2.resize(img, (width, height), interpolation=interp)
        cv2.imwrite((os.path.join(dest_dir, filename)).replace(src_fmt, dest_fmt), img_resized)


def draw_mask_contour(img, mask, color=(0, 255, 0)):
    """
    Draw mask contour on the image
    Args:
        img: base BGR image of shape (H, W, 3, dtype=uint8)
        mask: mask of shape (H, W, dtype=uint8)
        color: contour color in (B, G, R)
    Returns:
        BGR image of shape (H, W, 3, dtype=uint8)
    """
    _, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea)
    cv2.drawContours(img, contours, 0, color, thickness=1)
    return img


def filter_contour(mask):
    """
    Filter the mask to have single dominant contour region
    Args:
        mask:  mask of shape (H, W, dtype=uint8)
    Returns:
        mask with single contour
    """
    _, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = sorted(contours, key=cv2.contourArea)[-1]
    empty_img = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.uint8)
    # fill largest contour with white
    cv2.drawContours(empty_img, largest_contour, 0, (255,255,255), thickness=cv2.FILLED)
    assert empty_img.shape == mask.shape
    return empty_img


def draw_mask_layer(img, mask, color, alpha=0.5):
    """
    Draw a transparent mask layer on top of image
    Args:
        img: base BGR image of shape (H, W, 3, dtype=uint8)
        mask: binary mask of shape (H, W, dtype=uint8)
        color: contour color in (B, G, R)
        alpha: level of transperancy in [0,1]
    Returns:
        BGR image of shape (H, W, 3, dtype=uint8)
    """
    for c in range(3):
        found = img[:, :, c]*(1 - alpha) + alpha * color[c]
        not_found = img[:, :, c]
        img[:, :, c] = np.where(mask == 255, found, not_found)
    return img


def image_to_array(image_paths):
    """
    Read images specified by path into a array of RGB images
    Args:
        image_paths: list of image paths
    Returns:
        array of images in float RGB array format
    """
    images = []
    for image_path in image_paths:
        try:
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
            img = img / 255.
            images.append(img)
        except FileNotFoundError as e:
            print(e)
    return np.array(images)


def image_to_string(image_paths):
    """
    Reas images specified by path into a array of strings
    Args:
        image_paths: list of image paths or single image_path
    Returns:
        array of  images in string format
    """
    def convert_to_str(image_path):
        with open(image_path, 'rb') as f:
            return f.read()

    if type(image_paths) != str:
        img_str_list = list(map(lambda x: convert_to_str(x), image_paths)) # what happens to file read error?
        return np.array(img_str_list)
    else:
        return np.array(convert_to_str(image_paths))