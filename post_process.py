from skimage.measure import regionprops, label
import numpy as np


def post_process(segmentation):
    label_image = label(segmentation)
    regions = regionprops(label_image, intensity_image=segmentation)
    # Remove all objects below size 10x10
    for region in regions:
        if region.filled_area < 20*20:
            segmentation[label_image == region.label] = 0
            print('Removing region')

    # Only keep largest object for the nerves
    label_image = label(segmentation)
    regions = regionprops(label_image, intensity_image=segmentation)
    for label_id in range(2, 6):
        largest_area = 0
        largest_region = 0
        for region in regions:
            if int(region.max_intensity) != label_id:
                continue

            if region.filled_area > largest_area:
                largest_area = region.filled_area
                largest_region = region.label

        if largest_region > 0:
            segmentation[np.logical_and(segmentation == label_id, label_image != largest_region)] = 0

    return segmentation
