import os
from random import randint
from keras.callbacks import ModelCheckpoint
from dataset import get_dataset_files
from network import create_fc_network
import h5py
import numpy as np
from keras_tools.data_augmentation import UltrasoundImageGenerator
from loss import custom_fcn_loss, dice_loss, dice_loss_with_shape_model_regularization
from metrics import dice_metric
from keras.optimizers import SGD
from math import ceil
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
import keras.backend as K


def get_session(gpu_fraction=0.5):

    num_threads = os.environ.get('OMP_NUM_THREADS')
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)

    if num_threads:
        return tf.Session(config=tf.ConfigProto(
            gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
    else:
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

#KTF.set_session(get_session())

nr_of_objects = 6
path = '/home/smistad/data/eyeguide/axillary_nerve_block_annotated/'

experiments = [
    #'extra_deep',
    #'batch',
    #'flip',
    #'flip_gamma_rotate',
    #'flip_gamma_rotate_shadow_elastic',
    #'gamma_rotate',
    'flip',
    'rotate',
    'gamma',
    'shadow',
    'elastic',
    #'flip_gamma_rotate_batch',
    #'flip_gamma_rotate_shadow',
    #'flip_gamma_rotate_shadow_batch',
    #'flip_gamma_rotate_shadow_elastic_batch',
    #'flip_elastic_shadow',
    #'default',
    #'elastic_shadow',
    #'elastic_shadow_batch',
]

subjects = range(51)

def get_validation_set(test_subject, subjects_for_validation=3):
    exclude = [test_subject, 9, 43]
    validation_set = []
    for _ in range(subjects_for_validation):
        subject = randint(0, 50)
        while subject in exclude:
            subject = randint(0, 50)
        exclude.append(subject)
        validation_set.append(subject)
    return validation_set


for experiment_name in experiments:
    for subject_id in subjects:
        if subject_id == 9 or subject_id == 43:
            continue

        test_subject = subject_id
        validation_subjects = get_validation_set(test_subject)

        print('')
        print('Training with subjects ' + str(validation_subjects) + ' for validation ' + ' subject ' + str(test_subject) + ' for test')
        print('=================================================')

        extra_deep = False
        exclude_list = [test_subject, 9, 43]
        exclude_list.extend(validation_subjects)
        training_files = get_dataset_files(path, exclude=exclude_list)
        print('Training files: ', training_files)
        if experiment_name.find('batch') >= 0:
            training_generator = UltrasoundImageGenerator(training_files, all_files_in_batch=True)
        else:
            training_generator = UltrasoundImageGenerator(training_files)
        if experiment_name.find('gamma') >= 0:
            print('Gamma augmentation')
            training_generator.add_gamma_transformation()
        else:
            training_generator.add_intensity_scaling(0.9, 1.1)
        if experiment_name.find('rotate') >= 0:
            print('Rotate augmentation')
            training_generator.add_rotation(10)
        if experiment_name.find('elastic') >= 0:
            print('Elastic augmentation')
            training_generator.add_elastic_deformation(1024, 1024)
        if experiment_name.find('shadow') >= 0:
            print('Shadow augmentation')
            training_generator.add_gaussian_shadow()
        if experiment_name.find('shift') >= 0:
            print('Shifting')
            training_generator.add_shifting(25, 25, True)
        if experiment_name.find('flip') >= 0:
            print('Flipping')
            training_generator.add_flip()
        if experiment_name.find('extra_deep') >= 0:
            print('Extra deep network')
            extra_deep = True

        model = create_fc_network(nr_of_objects, input_shape=(256, 256, 1), extra_deep=extra_deep)
        model.compile(
            loss=dice_loss,
            optimizer='adam',
        )

        validation_files = get_dataset_files(path, only=validation_subjects)
        if len(validation_files) == 0:
            continue
        print('Validation files: ', validation_files)
        validation_generator = UltrasoundImageGenerator(validation_files)

        # Callback for saving best model
        save_best = ModelCheckpoint(
            'models/model_' + str(test_subject) + '_' + experiment_name + '.hdf5',
            monitor='val_loss',
            verbose=1,
            save_best_only=True,
            save_weights_only=True,
            mode='auto',
            period=1
        )

        if experiment_name.find('batch') >= 0:
            batch_size = 44
        else:
            batch_size = 32
        print('Batch size:', batch_size)
        model.fit_generator(
            training_generator.flow(batch_size=batch_size),
            steps_per_epoch=ceil(training_generator.get_size()/batch_size),
            epochs=75,
            validation_data=validation_generator.flow(batch_size=batch_size),
            validation_steps=ceil(validation_generator.get_size()/batch_size),
            callbacks=[save_best]
        )

        print('Finshed training')
        K.clear_session()
