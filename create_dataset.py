import os
from PIL import Image
import numpy as np
from skimage import transform
import h5py

# Backside arm
exclude_list = [
    140, 141, 159, 176, 184, 200, 207, 217, 218, 224, 273, 274, 335, 444, 518, 519
]

# Partially backside arm
exclude_list.extend([
    275, 283, 284, 286, 297, 305, 306, 324, 325, 326, 347, 348, 50
])

# Deformation (injection etc.)
exclude_list.extend([
    161, 162, 163, 167, 168, 169, 170, 174, 175, 182, 183, 188, 192, 193, 194, 195, 204, 205, 206, 208, 209, 210, 211, 212, 214, 215, 216, 221, 222, 223, 234, 235, 239, 241, 249,
    258, 259, 260, 467, 469, 470, 471, 472, 473, 474, 475, 476, 477, 480, 490, 491, 499, 500, 501, 558
])

def create_dataset(path, subject_list = None):
    input_width = 256
    input_height = 256
    output_width = 256
    output_height = 256

    # Open images folder, get all images
    for root, dirnames, _ in os.walk(path):
        for subject in dirnames:
            print(subject)
            images = []
            labels = []
            subject_path = os.path.join(root, subject)
            is_left = False
            if subject[-4:] == 'LEFT':
                is_left = True
                print('Recording is LEFT, flipping image and coordinates')
                subject_id = subject.split('_')[0]
            else:
                subject_id = subject

            print('Processing subject', subject_id)
            if subject_list is not None:
                if subject not in subject_list:
                    print('Skipping subject', subject)
                    continue
            for _, _, filenames in os.walk(subject_path):
                for filename in filenames:
                    if filename[-3:] != 'png':
                        continue
                    if int(filename[:-4]) in exclude_list:
                        print('Excluding ', filename[:-4])
                        continue
                    image_path = os.path.join(subject_path, filename)
                    image = Image.open(image_path)
                    #print(image_path)
                    pixels = np.array(image)
                    #print(pixels.shape)

                    # Resize image
                    width = pixels.shape[1]
                    height = pixels.shape[0]

                    scale = float(input_width) / width
                    image = pixels.astype(np.float32) / 255
                    transformed_image = transform.rescale(image, scale, preserve_range=True) # This also transforms image to be between 0 and 1
                    if transformed_image.shape[0] > input_height:
                        # Cut image
                        transformed_image = transformed_image[:input_height, :]
                    elif transformed_image.shape[0] < input_height:
                        transformed_image2 = np.zeros((input_height, input_width), dtype=np.float32)
                        transformed_image2[:transformed_image.shape[0], :] = transformed_image
                        transformed_image = transformed_image2

                    if is_left:
                        #print('Flipping image..')
                        #print('Subject: ', subject)
                        transformed_image = np.fliplr(transformed_image)

                    images.append(transformed_image)

                    # Read corresponding text file and data
                    # Get all boxes
                    label_path = os.path.join(subject_path, filename[:-3] + 'txt')
                    boxes = []
                    with open(label_path, 'r') as f:
                        for line in f:
                            line = line.replace('\n', '')
                            box = [int(d) for d in line.split(' ')]
                            boxes.append(box)

                    # For each label
                    data_y = np.zeros((output_height, output_width, 6))
                    data_y[:, :, 0] = 1 # Initialize all pixels to be background
                    for box in boxes:
                            center_x = box[1]*scale/input_width
                            center_y = box[2]*scale/input_height
                            if is_left:
                                # Flip x coordinate
                                center_x = 1 - center_x
                            size_x = box[3]*scale/input_width
                            size_y = box[4]*scale/input_height

                            # Convert to output pixel coordinates
                            center_x *= output_width
                            center_y *= output_height
                            size_x *= output_width
                            size_y *= output_height
                            center_x = int(center_x)
                            center_y = int(center_y)
                            max_x = center_x + max(1, round(size_x/2))
                            max_y = center_y + max(1, round(size_y/2))
                            min_x = center_x - int(round(size_x/2))
                            min_y = center_y - int(round(size_y/2))

                            # Mark all pixels inside box as 1.
                            # TODO How to handle overlap properly??
                            data_y[max(0, min_y):min(output_height, max_y), max(0, min_x):min(output_width, max_x), :] = 0 # Set all to zero first, to avoid overlap issues
                            label_id = box[0]
                            if label_id == 5:
                                label_id = 0
                            data_y[max(0, min_y):min(output_height, max_y), max(0, min_x):min(output_width, max_x), 1+label_id] = 1
                    labels.append(data_y)
            # Create a hdf5 file for each subject
            data_x = np.ndarray((len(images), input_height, input_width, 1))
            data_y = np.ndarray((len(images), output_height, output_width, 6))
            for i in range(len(images)):
                data_x[i, :, :, 0] = images[i]
                data_y[i, :, :, :] = labels[i]
            f = h5py.File(os.path.join(path, subject_id + '.hd5'), 'w')
            f.create_dataset("data", data=data_x, compression="gzip", compression_opts=4)
            f.create_dataset("label", data=data_y, compression="gzip", compression_opts=4)
            f.close()
            print('Finished creating hdf5 dataset for subject', subject_id)



# Import data
path = '/home/smistad/data/eyeguide/axillary_nerve_block_annotated/'

#create_dataset(path)
print('Excluded in total:', len(exclude_list))

