from keras.models import load_model
from dataset import get_dataset_files
from keras_tools.data_augmentation import UltrasoundImageGenerator
import matplotlib.pyplot as plt
import numpy as np
import keras.backend as K
from math import sqrt, floor
from loss import dice_loss

# Functions for getting activations of a layer, given a bathc with inputs
def get_activations(model, layer, X_batch):
    get_activations = K.function([model.layers[0].input, K.learning_phase()], [model.get_layer(layer).output,])
    activations = get_activations([X_batch,0])
    return activations

# Load model and setup generator
model = load_model('models/shape_model.hd5', custom_objects={'dice_loss': dice_loss})
files = get_dataset_files('shape-model-dataset/', only=[1])
generator = UltrasoundImageGenerator(files)
#generator.add_shifting(25, 25)
generator.add_label_flipping(range(6))

counter = 0
code_length = 64
data = np.ndarray((code_length, generator.get_size()))  # Accumulate activation data from low dimensional shape model
for input, output in generator.flow(batch_size=32):
    predicted_output = model.predict_on_batch(input)
    hidden_activations = get_activations(model, 'low_dimensional_shape_model', input)[0]
    for i in range(input.shape[0]):
        data[:, counter] = hidden_activations[i, :]
        f, axes = plt.subplots(2, 5, figsize=(20, 20))
        for j in range(1,6):
            axes[0, j-1].imshow(input[i, :, :, j])
            axes[1, j-1].imshow(predicted_output[i, :, :, j])
        plt.show()

        counter += 1
    if counter >= generator.get_size():
        break

# Plot histograms of activations for low dimensional shape model
rows = int(floor(sqrt(code_length)))
cols = rows
f, axes = plt.subplots(cols, rows, figsize=(20, 20))
for x in range(cols):
    for y in range(rows):
        axes[x, y].hist(data[x + y*cols, :], bins=10)
plt.show()