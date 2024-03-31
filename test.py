from dataset import get_dataset_files
from keras_tools.data_augmentation import UltrasoundImageGenerator
from keras_tools.visualization import display_convolutions
from network import create_fc_network
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

nr_of_objects = 6

path = '/home/smistad/data/eyeguide/axillary_nerve_block_annotated/'
files = get_dataset_files(path, only=[0, 1, 2])
generator = UltrasoundImageGenerator(files)
generator2 = UltrasoundImageGenerator(files)
#generator.add_elastic_deformation(1024, 1024)
generator.add_flip()
generator.add_gamma_transformation()
generator.add_rotation(10)
#generator.add_gaussian_shadow()
#generator.add_shifting(25, 25, True)
#generator.add_intensity_scaling(0.9, 1.1)

show_segmentation = True
single_image = False

for input, output in generator.flow(batch_size=64, shuffle=False):
    for input2, output2 in generator2.flow(batch_size=64, shuffle=False):
         for j in range(input2.shape[0]):
             if single_image:
                 plt.figure(figsize=(10, 10))
                 plt.axis('off')
                 plt.imshow(input[j, :, :, 0], cmap='gray', vmin=0, vmax=1)
                 plt.imshow(input2[j, :, :, 0], cmap='inferno', alpha=0.5, vmin=0, vmax=1)
                 plt.show()
                 continue

             if show_segmentation:
                 f, axes = plt.subplots(1, 8, figsize=(20, 10))
             else:
                 f, axes = plt.subplots(1, 2, figsize=(20, 10))

             axes[0].imshow(input2[j, :, :, 0], cmap='gray', vmin=0, vmax=1)
             axes[0].axis('off')
             axes[1].imshow(input[j, :, :, 0], cmap='gray', vmin=0, vmax=1)
             axes[1].axis('off')
             if show_segmentation:
                 axes[2].imshow(output[j, :, :, 0], cmap='gray', vmin=0, vmax=1)
                 axes[2].axis('off')
                 axes[3].imshow(output[j, :, :, 1], cmap='gray', vmin=0, vmax=1)
                 axes[3].axis('off')
                 axes[4].imshow(output[j, :, :, 2], cmap='gray', vmin=0, vmax=1)
                 axes[4].axis('off')
                 axes[5].imshow(output[j, :, :, 3], cmap='gray', vmin=0, vmax=1)
                 axes[5].axis('off')
                 axes[6].imshow(output[j, :, :, 4], cmap='gray', vmin=0, vmax=1)
                 axes[6].axis('off')
                 axes[7].imshow(output[j, :, :, 5], cmap='gray', vmin=0, vmax=1)
                 axes[7].axis('off')
             plt.show()