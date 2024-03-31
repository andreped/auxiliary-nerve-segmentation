from math import ceil
from tensorflow.python.keras.callbacks import ModelCheckpoint, Callback, EarlyStopping, ReduceLROnPlateau
import os
from batch_generator import *
from keras_tools.network import Unet


def finder(test, path):
    sets = []
    for l in os.listdir(path):
        for t in test:
            if l.startswith(t):
                sets.append(path + l)
    return sets

def import_dataset(tmp, path, name_curr):
    file = h5py.File(path + 'dataset_' + name_curr + '.h5', 'r')
    tmp = np.array(file[tmp])
    tmp = [tmp[i].decode("UTF-8") for i in range(len(tmp))]
    file.close()
    return tmp


#if __name__ == "__main__":


### Set Training params

# use single GPU (first one)
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# set name for model
name = "17_07_multi_seg_7_classes_spat_drop_0.1_deepest_bs_32_zoom_aug_adam_decay_stale_50_2000_epochs_3"

# number of classes
num_classes = 7

# paths
path = "/home/andrep/anesthesia/"
data_path = path + "data/anesthesia_seg_12_07/"
save_model_path = path + "output/models/"
history_path = path + "output/history/"
datasets_path = path + "output/datasets/"

# Option 1: generate random sets by randomly assigning k patients to train, val and test set
locs = []
pats = []
for l in os.listdir(data_path):
    pats.append(l[:3])
pats = np.unique(pats)

k = 7
shuffle(pats)
test = pats[:k]
val = pats[k:int(2 * k)]
train = pats[int(2 * k):]

test_set = finder(test, data_path)
val_set = finder(val, data_path)
train_set = finder(train, data_path)

# option 2: use fixed set
#path_curr = "15_07_multi_seg_7_classes_spat_drop_0.2_deepest_bs_8" # <- use the same split to evaluate methods
path_curr = "16_07_multi_seg_7_classes_spat_drop_0.2_deepest_bs_8_zoom_aug_adam"

test_set = import_dataset('test', datasets_path, path_curr)
val_set = import_dataset('val', datasets_path, path_curr)
train_set = import_dataset('train', datasets_path, path_curr)


# print percentage split
NN = len(test_set) + len(val_set) + len(train_set)
print("Train: " + str(len(train_set)/NN))
print("Val: " + str(len(val_set)/NN))
print("Test: " + str(len(test_set)/NN))


# save generated data sets
f = h5py.File((datasets_path + 'dataset_' + name + '.h5'), 'w')
f.create_dataset("test", data=np.array(test_set).astype('S200'), compression="gzip", compression_opts=4)
f.create_dataset("val", data=np.array(val_set).astype('S200'), compression="gzip", compression_opts=4)
f.create_dataset("train", data=np.array(train_set).astype('S200'), compression="gzip", compression_opts=4)
f.close()

# define model
network = Unet(input_shape=(256, 256, 1), nb_classes=num_classes)
network.encoder_spatial_dropout = 0.1
network.decoder_spatial_dropout = 0.1
#network.set_convolutions([4, 8, 16, 32, 64, 128, 256, 512, 256, 128, 64, 32, 16, 8, 4])
network.set_convolutions([8, 16, 32, 64, 128, 256, 512, 256, 128, 64, 32, 16, 8])
model = network.create()

print(model.summary())

#
model.compile(
    #optimizer='adadelta',
    optimizer='adam',
    loss=network.get_dice_loss()
)

# hyperpar
batch_size = 32
epochs = 2000

# augmentation
train_aug = {'flip': 1, 'gamma': [0.25, 1.75], 'rotate': 10, 'affine': [0.8, 1.0, 0.08, 0.1],\
             'shadow': [0.1*256, 0.9*256, 0.1*256, 0.9*256, 0.25, 0.8]} # <- Erik's augs
train_aug = {'flip': 1, 'gamma': [0.25, 1.75], 'rotate': 10, 'affine': [0.8, 1.0, 0.08, 0.1],\
             'shadow': [0.1*256, 0.9*256, 0.1*256, 0.9*256, 0.25, 0.8],
             'scale':[0.75, 1.5]} # <--- added zoom aug
val_aug = {}

train_gen = batch_gen2(train_set, batch_size, aug=train_aug, num_classes=num_classes, epochs=epochs)
val_gen = batch_gen2(val_set, batch_size, aug=val_aug, num_classes=num_classes, epochs=epochs)

'''
train_gen = UltrasoundImageGenerator(train_set)
#training_generator.add_gamma_transformation(0.75, 1.25)
#training_generator.add_shifting(20, 20, preserve_segmentation=True)
#training_generator.add_flip()
#training_generator.add_rotation(180)
#training_generator.add_scaling()

val_gen = UltrasoundImageGenerator(val_set)
#validation_generator.add_flip()
#validation_generator.add_rotation(180)
#validation_generator.add_scaling()
'''

train_length = len(train_set)
val_length = len(val_set)

# Reduce learning rate on plateau
lr_reduce_plateau = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=50,
)

# Early stopping
early_stopping_callback = EarlyStopping(
    monitor='val_loss',
    patience=1000,
)

save_best = ModelCheckpoint(
    save_model_path + "model_" + name + ".h5",
    monitor='val_loss',
    verbose=1,
    save_best_only=True,
    save_weights_only=False,
    mode='auto',
    period=1
)

class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.losses.append(['loss', 'val_loss'])

    def on_epoch_end(self, batch, logs={}):
        self.losses.append([logs.get('loss'), logs.get('val_loss')])
        # save history:
        f = h5py.File((history_path + 'history_' + name + '.h5'), 'w')
        f.create_dataset("history", data=np.array(self.losses).astype('|S9'), compression="gzip", compression_opts=4)
        f.close()

history_log = LossHistory()

history = model.fit_generator(
    train_gen,
    steps_per_epoch=int(ceil(train_length / batch_size)),  # <- NO ceil(int()) in Keras2-API?
    epochs=epochs,
    validation_data=val_gen,
    validation_steps=int(ceil(val_length / batch_size)),
    callbacks=[save_best, history_log, early_stopping_callback, lr_reduce_plateau]
    # use_multiprocessing=True,
    # workers = 1
)
