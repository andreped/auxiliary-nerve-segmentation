from math import isnan
from skimage.measure import regionprops
from skimage.morphology import dilation, square
from skimage.segmentation import find_boundaries
from dataset import get_dataset_files
from keras_tools.data_augmentation import UltrasoundImageGenerator
from keras_tools.visualization import display_convolutions
from network import create_fc_network
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
from PIL import Image
import numpy as np
from skimage import transform
from os.path import join
from create_dataset import exclude_list
from numpy.lib.stride_tricks import as_strided

from plot import create_result_image
from post_process import post_process


def get_data(path, subject_list=None):
    input_width = 256
    input_height = 256
    output_width = 256
    output_height = 256
    result = []

    # Open images folder, get all images
    for root, dirnames, _ in os.walk(path):
        for subject in dirnames:
            subject_path = os.path.join(root, subject)
            is_left = False
            if subject[-4:] == 'LEFT':
                is_left = True
                #print('Recording is LEFT, flipping image and coordinates')
                subject_id = subject.split('_')[0]
            else:
                subject_id = subject

            #print('Processing subject', subject_id)
            if subject_list is not None:
                if int(subject_id) not in subject_list:
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
                        image = np.fliplr(image)

                    # Read corresponding text file and data
                    # Get all boxes
                    label_path = os.path.join(subject_path, filename[:-3] + 'txt')
                    boxes = []
                    with open(label_path, 'r') as f:
                        for line in f:
                            line = line.replace('\n', '')
                            box = [int(d) for d in line.split(' ')]
                            if box[0] == 5:
                                box[0] = 0
                            # Remove duplicate boxes
                            box_duplicate = False
                            for box2 in boxes:
                                equal = True
                                for i in range(len(box)):
                                    if box[i] != box2[i]:
                                        equal = False
                                        break
                                if equal:
                                    box_duplicate = True
                                    break
                            if not box_duplicate:
                                boxes.append(box)

                    # For each label
                    new_boxes = []
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
                        new_boxes.append([box[0], min_x, min_y, max_x, max_y])
                        data_y[max(0, min_y):min(output_height, max_y), max(0, min_x):min(output_width, max_x), :] = 0 # Set all to zero first, to avoid overlap issues
                        data_y[max(0, min_y):min(output_height, max_y), max(0, min_x):min(output_width, max_x), 1+box[0]] = 1

                    result.append({
                        'image': transformed_image,
                        'boxes': new_boxes,
                        'segmentation': data_y,
                        'name': filename,
                        'original_image': image,
                        'scale': scale,
                    })
    return result

experiments = [
    #'default',
    'flip_gamma_rotate_shadow_elastic',
    #'flip_gamma_rotate_batch',
]
for experiment_name in experiments:
    os.makedirs(join('results', experiment_name, 'segmentation_images'), exist_ok=True)
    subjects = range(51)
    visualize = True
    subject_summary = False
    confidence_threshold = 0.5
    confidence_threshold2 = 0.5
    nr_of_objects = 6
    path = '/home/smistad/data/eyeguide/axillary_nerve_block_annotated/'
    total_true_positive = [0, 0, 0, 0, 0]
    total_false_positive = [0, 0, 0, 0, 0]
    total_false_negative = [0, 0, 0, 0, 0]
    true_positives = np.zeros((len(subjects), 5), dtype=np.int32)
    false_positives = np.zeros((len(subjects), 5), dtype=np.int32)
    false_negatives = np.zeros((len(subjects), 5), dtype=np.int32)
    precision_per_subject_and_object = np.ndarray((len(subjects), 5), dtype=np.float32)
    precision_per_subject_and_object[:, :] = np.nan
    recall_per_subject_and_object = np.ndarray((len(subjects), 5), dtype=np.float32)
    recall_per_subject_and_object[:, :] = np.nan
    examples = np.zeros((len(subjects)), dtype=np.int32)
    counter = 0

    model = create_fc_network(nr_of_objects, input_shape=(256, 256, 1))

    subject_counter = 0
    result_images = []
    for subject_id in subjects:
        if subject_id == 9 or subject_id == 43:
            subject_counter += 1
            continue
        data = get_data(path, subject_list=[subject_id])
        print('Calculating accuracy for ', subject_id, 'with', len(data), ' examples')
        model.load_weights('models/model_' + str(subject_id) + '_' + experiment_name + '.hdf5')
        for example in data:
            predicted_output = model.predict(np.reshape(example['image'], (1, 256, 256, 1)))


            counter += 1
            examples[subject_counter] += 1

            # Go through each box
            true_positive = [0, 0, 0, 0, 0]
            false_positive = [0, 0, 0, 0, 0]
            false_negative = [0, 0, 0, 0, 0]
            predicted_segmentation = np.zeros([256, 256], dtype=np.uint8)
            for i in range(1, 6):
                predicted_segmentation[predicted_output[0, :, :, i] > confidence_threshold] = i

            original_predicted_segmentation = np.copy(predicted_segmentation)
            #predicted_segmentation = post_process(predicted_segmentation)

            for box in example['boxes']:
                label_id, min_x, min_y, max_x, max_y = box
                object_is_detected = np.sum(predicted_segmentation == label_id + 1) > 0
                box_image = np.zeros([256, 256], dtype=np.uint8)
                box_image[max(0, min_y):min(256, max_y), max(0, min_x):min(256, max_x)] = 1
                if object_is_detected:
                    # If segmentation covers over 25% of the box it is a true_positive
                    object_segmentation = np.zeros([256, 256], dtype=np.uint8)
                    object_segmentation[predicted_segmentation == label_id + 1] = 1
                    box_area = np.sum(box_image)
                    print('Object:', label_id)
                    print('Coverage is:', np.sum(box_image*object_segmentation)/box_area)
                    if np.sum(box_image * object_segmentation) >= box_area*0.25:
                        true_positive[label_id] += 1
                        # TODO remove entire region from segmentation
                        # Remove this box from segmentation, but first dilate it
                        box_image = dilation(box_image, square(9))
                        predicted_segmentation[box_image == 1] = 0
                    else:
                        # If it covers less it is false_negative
                        false_negative[label_id] += 1
                        # Remove this box from segmentation, but first dilate it
                        box_image = dilation(box_image, square(9))
                        predicted_segmentation[box_image == 1] = 0
                else:
                    false_negative[label_id] += 1

            # Count false positives
            regions = regionprops(predicted_segmentation)
            for label_id in range(5):
                # Count nr of "islands" in the background class of the true segmentation for each label
                for region in regions:
                    if region.label != label_id + 1:
                        continue
                    print('Object:', region.label-1)
                    print('Area:', region.filled_area)
                    if region.filled_area > 20*20: # Limit on minimum size of detect object
                        false_positive[label_id] += 1

            create_result_image(example, original_predicted_segmentation, true_positive, false_positive, false_negative, filename=join('results', experiment_name, 'segmentation_images', str(subject_id) + '_' + str(counter) + '.png'))
            print('New example')
            print('=====================')
            for j in range(5):  # classes, excluding background
                print('For object ', j)
                print('True positives', true_positive[j])
                print('False positives', false_positive[j])
                print('False negatives', false_negative[j])
                print('')

                total_true_positive[j] += true_positive[j]
                total_false_positive[j] += false_positive[j]
                total_false_negative[j] += false_negative[j]

                true_positives[subject_counter, j] += true_positive[j]
                false_positives[subject_counter, j] += false_positive[j]
                false_negatives[subject_counter, j] += false_negative[j]

            precision = np.sum(true_positive) / (np.sum(true_positive) + np.sum(false_positive))
            recall = np.sum(true_positive) / (np.sum(true_positive) + np.sum(false_negative))
            f_measure = 2.0*recall*precision/(recall + precision)
            if isnan(f_measure):
                f_measure = 0
            result_images.append({
                'subject': subject_id,
                'name': example['name'],
                'f_measure': f_measure,
            })

            if visualize:
                plt.show()
        for j in range(5):
            precision_per_subject_and_object[subject_counter, j] = true_positives[subject_counter, j] / (true_positives[subject_counter, j] + false_positives[subject_counter, j])
            recall_per_subject_and_object[subject_counter, j] = true_positives[subject_counter, j] / (true_positives[subject_counter, j] + false_negatives[subject_counter, j])
            if false_negatives[subject_counter, j] + true_positives[subject_counter, j] == 0:
                # Object does not exist in this subjects data
                # Do not do statistics on this subject and object, therefore set to nan
                precision_per_subject_and_object[subject_counter, j] = np.nan
                recall_per_subject_and_object[subject_counter, j] = np.nan
        subject_counter += 1


    print('Nr of annotated examples: ', len(result_images))

    # # Sort results using f-measure
    # result_images = sorted(result_images, key=lambda result: result['f_measure'])
    # # Median case
    # median_score = result_images[len(result_images)//2]['f_measure']
    # counter = 0
    # for result in result_images:
    #     if result['f_measure'] == median_score:
    #         plt.figure()
    #         plt.imshow(result['image'])
    #         plt.savefig(join('results', experiment_name, 'median_case_' + str(counter) + '.png'))
    #         plt.close()
    #         counter += 1
    #
    # # Worst case
    # counter = 0
    # for result in result_images:
    #     if result['f_measure'] == 0:
    #         plt.figure()
    #         plt.imshow(result['image'])
    #         plt.savefig(join('results', experiment_name, 'worst_case_' + str(counter) + '.png'))
    #         plt.close()
    #         counter += 1

    print('Total Summary')
    print('=========================')
    print('Examples processed:', counter)
    print('=========================')
    recall = np.zeros((5))
    precision = np.zeros((5))
    for j in range(5):  # classes, excluding background
        print('Object:', j+1)
        print('------------------------')
        print('True positive', total_true_positive[j])
        print('False positive', total_false_positive[j])
        print('False negative', total_false_negative[j])
        recall[j] = round(total_true_positive[j]/(total_true_positive[j] + total_false_negative[j]), 2)
        precision[j] = round(total_true_positive[j]/(total_true_positive[j] + total_false_positive[j]), 2)
        print('Precision (percentage of correct findings)', precision[j])
        print('Recall (percentage of actual objects found)', recall[j])
        print('')

    def autolabel(rects):
        """
        Attach a text label above each bar displaying its height
        """
        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width(), 1.02*height,
                    '%.2f' % height,
                    ha='center', va='bottom')

    fig, ax = plt.subplots()
    width = 0.35
    indexes = np.arange(5)
    precision_per_object = np.ndarray((5,))
    precision_std_per_object = np.ndarray((5,))
    recall_per_object = np.ndarray((5,))
    recall_std_per_object = np.ndarray((5,))
    for i in range(5):
        precision_per_object[i] = precision_per_subject_and_object[~np.isnan(precision_per_subject_and_object[:, i]), i].mean()
        precision_std_per_object[i] = precision_per_subject_and_object[~np.isnan(precision_per_subject_and_object[:, i]), i].std()
        recall_per_object[i] = recall_per_subject_and_object[~np.isnan(recall_per_subject_and_object[:, i]), i].mean()
        recall_std_per_object[i] = recall_per_subject_and_object[~np.isnan(recall_per_subject_and_object[:, i]), i].std()
    p1 = ax.bar(indexes, recall_per_object, width, yerr=recall_std_per_object, color='blue')
    p2 = ax.bar(indexes + width, precision_per_object, width, yerr=precision_std_per_object, color='red')
    #plt.xlabel('Object')
    ax.set_xticks(indexes + width / 2)
    ax.set_xticklabels(('Blood vessel', 'MSC', 'Median', 'Ulnar', 'Radial'))
    plt.legend((p1[0], p2[0]), ('Recall', 'Precision'))
    plt.ylim([0, 1])
    plt.yticks(np.arange(0, 1.1, 0.1))
    autolabel(p1)
    autolabel(p2)
    plt.savefig(join('results', experiment_name, 'precision_recall.png'), bbox_inches='tight')
    plt.show()

    print('Means recall: ', recall_per_object)
    print('Means precision: ', precision_per_object)
    print('Std recall: ', recall_std_per_object)
    print('Std precision: ', precision_std_per_object)

    subject_recall = np.ndarray((len(subjects), 5))
    if subject_summary:
        print('Subject Summary')
        print('=========================')
        for subject_counter in range(len(subjects)):
            if subject_counter in [9, 43]:
                continue
            print('Subject ID:', subjects[subject_counter])
            print('=========================')
            print('Examples processed:', examples[subject_counter])
            print('=========================')
            for j in range(5):  # classes, excluding background
                if true_positives[subject_counter, j] + false_negatives[subject_counter, j] == 0:
                    # Object is not in ground truth
                    subject_recall[subject_counter, j] = 0
                    continue
                print('Object:', j+1)
                print('------------------------')
                print('True positive', true_positives[subject_counter, j])
                print('False positive', false_positives[subject_counter, j])
                print('False negative', )
                subject_recall[subject_counter, j] = round(true_positives[subject_counter, j]/(true_positives[subject_counter, j] + false_negatives[subject_counter, j]), 2)
                print('Precision (percentage of correct findings)', round(true_positives[subject_counter, j]/(true_positives[subject_counter, j] + false_positives[subject_counter, j]), 2))
                print('Recall (percentage of actual objects found)', subject_recall[subject_counter, j])
                print('')


        fig, ax = plt.subplots(figsize=(20, 10))
        width = 0.3
        indexes = np.arange(len(subjects))
        p1 = ax.bar(indexes, subject_recall[:, 0], width, color='red')
        p2 = ax.bar(indexes, subject_recall[:, 1], width, bottom=subject_recall[:, 0], color='yellow')
        p3 = ax.bar(indexes, subject_recall[:, 2], width, bottom=subject_recall[:, 0]+subject_recall[:, 1], color='green')
        p4 = ax.bar(indexes, subject_recall[:, 3], width, bottom=subject_recall[:, 0]+subject_recall[:, 1]+subject_recall[:, 2], color='purple')
        p5 = ax.bar(indexes, subject_recall[:, 4], width, bottom=subject_recall[:, 0]+subject_recall[:, 1]+subject_recall[:, 2]+subject_recall[:, 3], color='blue')
        plt.xlabel('Subject')
        plt.suptitle('Recall for each object and subject')
        ax.set_xticks(indexes)
        ax.set_xticklabels(subjects)
        plt.legend((p1[0], p2[0], p3[0], p4[0], p5[0]), ('Blood vessel', 'MSC', 'Median', 'Ulnar', 'Radial'))
        plt.savefig(join('results', experiment_name, 'recall_per_subject.png'), bbox_inches='tight')
        plt.show()

    # Median, worst case etc.
    new_array = np.copy(recall_per_subject_and_object)
    new_array[np.isnan(new_array)] = 0
    recall_per_subject = np.sum(new_array, axis=1)
    new_array = np.copy(precision_per_subject_and_object)
    new_array[np.isnan(new_array)] = 0
    precision_per_subject = np.sum(new_array, axis=1)
    f_score_per_subject = 2*recall_per_subject*precision_per_subject/(recall_per_subject + precision_per_subject)
    f_score_per_subject[np.isnan(f_score_per_subject)] = 0
    best_subject = np.argmax(f_score_per_subject)
    print('Best case: ', best_subject, np.max(f_score_per_subject))
    print('Median', np.median(f_score_per_subject))
    print('Median case:', np.argmax(f_score_per_subject == np.median(f_score_per_subject)))
    f_score_per_subject[f_score_per_subject == 0] = 10
    print('Worst case: ', np.argmin(f_score_per_subject), np.min(f_score_per_subject))
    for i in range(len(f_score_per_subject)):
        print(i, f_score_per_subject[i])

