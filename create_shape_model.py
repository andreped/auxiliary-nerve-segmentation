from loss import dice_loss

create_dataset = False
visualize = False
size = (256, 256)
path = '/home/smistad/data/eyeguide/axillary_nerve_block_annotated/'

from os.path import join
import matplotlib.pyplot as plt
import PIL
import numpy as np
import h5py
from autoencoder import create_autoencoder
from keras_tools.data_augmentation import UltrasoundImageGenerator
from dataset import get_dataset_files
from math import floor

# First create dataset for autoencoder
if create_dataset:
    all_files = get_dataset_files(path)
    for file in all_files:
        # TODO for each patient, open existing hdf5 files created with create_dataset
        # Copy out segmentation part
        print(file)
        patient_id = file[file.rfind('/') + 1:file.rfind('.')]
        print(patient_id)
        f = h5py.File(file, 'r')
        data = f['label']

        # Write to hdf5
        f2 = h5py.File('shape-model-dataset/' + str(patient_id) + '.hd5', 'w')
        f2.create_dataset("data", data=data, compression="gzip", compression_opts=4)
        f2.create_dataset("label", data=data, compression="gzip", compression_opts=4)
        f.close()
        f2.close()

# Setup autoencoder
model = create_autoencoder((size[0], size[1], 6), code_length=128)
model.compile(
    #loss='binary_crossentropy',    # This loss function converges against all background, class imbalance problem?
    loss=dice_loss,
    optimizer='adam',
)

training_files = get_dataset_files('shape-model-dataset/', exclude=[1])
training_generator = UltrasoundImageGenerator(training_files)
#training_generator.add_shifting(25, 25)
training_generator.add_label_flipping(range(6))
validation_files = get_dataset_files('shape-model-dataset/', only=[1])
validation_generator = UltrasoundImageGenerator(validation_files)
#validation_generator.add_shifting(25, 25)
validation_generator.add_label_flipping(range(6))

model.fit_generator(
    training_generator.flow(batch_size=32),
    steps_per_epoch=floor(training_generator.get_size()/32),
    epochs=40,
    validation_data=validation_generator.flow(batch_size=32),
    validation_steps=floor(validation_generator.get_size()/32),
)

model.save('models/shape_model.hd5')
model.save_weights('models/shape_model_weights.hd5')