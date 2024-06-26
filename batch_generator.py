# batch gen
import random
import h5py
import numpy as np
from scipy.ndimage.interpolation import rotate, shift, affine_transform, zoom
from numpy.random import random_sample, rand, random_integers, uniform
# import matplotlib.pyplot as plt
import cv2
import scipy
from numpy.random import shuffle
import numba as nb
import PIL
from io import BytesIO
from scipy.ndimage.interpolation import map_coordinates


# have to do this to use matplotlib viewing with GPU for some reason (used to work before, thus didn't need to do this...)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.switch_backend("TkAgg")


# apply random gaussian noise to RGB image
def add_gaussian_blur2(input_im, output, sigmas_max):
    #blur = cv2.GaussianBlur(img, (5,5), 0)

    # oscillate around 0 => add or substract value with equal probability of add/sub, and as strong in both directions
    means = (0, 0, 0)

    # random sigma uniform [0, sigmas_max]
    sigmas = (round(uniform(0, sigmas_max[0])), round(uniform(0, sigmas_max[1])), round(uniform(0, sigmas_max[2])))

    # RGB -> HSV
    input_im = cv2.cvtColor(input_im.astype(np.uint8), cv2.COLOR_RGB2HSV)

    # apply random sigma gaussian blur
    input_im = np.clip(input_im.astype(np.float32) + cv2.randn(input_im.astype(np.float32), means, sigmas), a_min=0, a_max=255).astype(np.uint8)

    # HSV -> RGB
    input_im = cv2.cvtColor(input_im, cv2.COLOR_HSV2RGB)

    return input_im, output


# quite slow -> Don't use this! Not optimized and doesn't fit our problem!
def add_affine_transform2(input_im, output, params):

    #params
    a1, a2, s1, s2 = params
    #a1, a2 = alpha_lims
    #s1, s2 = sigma_lims
    random_state = None

    #start = time.time()
    original_shape = input_im.shape
    original_output_shape = output.shape
    image = input_im[:, :, 0].astype(np.float32)
    # assert len(image.shape) == 2

    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    sigma = np.random.uniform(s1, s2) * shape[1]  # np.random.uniform(shape[0] * 0.5, shape[1] * 0.5), 0.08, 0.1
    alpha = np.random.uniform(a1, a2) * shape[1] # 0.8, 1.0
    # print('Parameters: ', alpha, sigma)

    blur_size = int(4 * sigma) | 1
    dx = cv2.GaussianBlur((random_state.rand(*shape) * 2 - 1), ksize=(blur_size, blur_size), sigmaX=sigma) * alpha
    dy = cv2.GaussianBlur((random_state.rand(*shape) * 2 - 1), ksize=(blur_size, blur_size), sigmaX=sigma) * alpha
    # dx = gaussian_filter(, sigma, mode="constant", cval=0) * alpha
    # dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dz = np.zeros_like(dx)

    # x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    # indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)), np.reshape(z, (-1, 1))
    x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
    indices = np.reshape(x + dx, (-1, 1)), np.reshape(y + dy, (-1, 1))

    # Transform output
    output[:, :, 0] = np.ones(output.shape[:2])  # Clear background
    for label in range(1, output.shape[2]):
        # segmentation = cv2.remap(output[:, :, label], dx, dy, interpolation=cv2.INTER_LINEAR).reshape(output.shape[:2])
        segmentation = map_coordinates(output[:, :, label], indices, order=0).reshape(output.shape[:2])
        output[:, :, label] = segmentation

    # Remove segmentation from other labels
    for label in range(1, output.shape[2]):
        for label2 in range(output.shape[2]):
            if label2 == label:
                continue
            output[output[:, :, label] == 1, label2] = 0

    # input = cv2.remap(image, dx, dy, interpolation=cv2.INTER_LINEAR).reshape(original_shape)
    input_im = map_coordinates(image, indices, order=1).reshape(original_shape)
    #end = time.time()

    # print('Elastic deformation time', (end - start))

    return input_im, output


def add_gauss_shadow2(input_im, output, params):
    sigma_x_min, sigma_x_max, sigma_y_min, sigma_y_max, strength_min, strength_max = params

    size = input_im.shape[0:2]
    for channel in [0]:#input_image.shape[3]:
        x, y = np.meshgrid(np.linspace(-1, 1, size[1], dtype=np.float32), np.linspace(-1, 1, size[0], dtype=np.float32), copy=False)
        x_mu = np.random.uniform(-1.0, 1.0, 1).astype(np.float32)
        y_mu = np.random.uniform(-1.0, 1.0, 1).astype(np.float32)
        sigma_x = np.random.uniform(sigma_x_min, sigma_x_max, 1).astype(np.float32)
        sigma_y = np.random.uniform(sigma_y_min, sigma_y_max, 1).astype(np.float32)
        strength = np.random.uniform(strength_min, strength_max, 1).astype(np.float32)
        g = 1.0 - strength * np.exp(-((x - x_mu) ** 2 / (2.0 * sigma_x ** 2) + (y - y_mu) ** 2 / (2.0 * sigma_y ** 2)), dtype=np.float32)

        input_im[:, :, channel] = input_im[:, :, channel]*np.reshape(g, size)

    return input_im, output


"""
###
input_im:		input image, 5d ex: (1,64,256,256,1) , (dimi0, z, x, y, channel)
output:			ground truth, 5d ex: (1,64,256,256,2), (dimi0, z, x, y, channel)
max_shift:		the maximum amount th shift in a direction, only shifts in x and y dir
###
"""

def add_shift2(input_im, output, max_shift):
    # randomly choose which shift to set for each axis (within specified limit)
    sequence = [round(uniform(-max_shift, max_shift)), round(uniform(-max_shift, max_shift)), 0]

    # apply shift to RGB-image
    input_im = shift(input_im.copy(), sequence, order=0, mode='constant', cval=0) # <- pad with "white"

    output[..., 0] = shift(output.copy()[..., 0], sequence[:-1], order=0, mode='constant', cval=1)
    output[..., 1:] = shift(output.copy()[..., 1:], sequence, order=0, mode='constant', cval=0)

    return input_im, output


"""
####
input_im:		input image, 5d ex: (1,64,256,256,1) , (dimi0, z, x, y, channel)
output:			ground truth, 5d ex: (1,64,256,256,2), (dimi0, z, x, y, channel)
min/max_angle: 	minimum and maximum angle to rotate in deg, positive integers/floats.
####
"""


# -> Only apply rotation in image plane -> faster and unnecessairy to rotate xz or yz

def add_rotation2(input_im, output, max_angle):
    # randomly choose how much to rotate for specified max_angle
    angle_xy = np.random.randint(-max_angle, max_angle)

    ## rotate image
    input_im = scipy.ndimage.rotate(input_im, angle_xy, order=1, reshape=False)  # Using order=2 here gives incorrect results

    ## rotate GT
    # Transform output
    output[:, :, 0] = np.ones(output.shape[:2])  # Clear background
    for label in range(1, output.shape[2]):
        segmentation = scipy.ndimage.rotate(output[:, :, label], angle_xy, order=0, reshape=False).reshape(
            output.shape[:2])
        output[:, :, label] = segmentation

    # Remove segmentation from other labels
    for label in range(1, output.shape[2]):
        for label2 in range(output.shape[2]):
            if label2 == label:
                continue
            output[output[:, :, label] == 1, label2] = 0

    return input_im, output


"""
flips the array along random axis, no interpolation -> super-speedy :)
"""


def add_flip2(input_im, output):
    # randomly choose whether or not to flip
    if (random_integers(0, 1) == 1):
        # randomly choose which axis to flip against
        #flip_ax = random_integers(0, high=1)
        flip_ax = 1

        # flip CT-chunk and corresponding GT
        input_im = np.flip(input_im, flip_ax)
        output = np.flip(output, flip_ax)

    return input_im, output


"""
performs intensity transform on the chunk, using gamma transform with random gamma-value
"""

def add_gamma2(input_im, output, r_limits):
    # limits
    r_min, r_max = r_limits

    # randomly choose gamma factor
    r = uniform(r_min, r_max)

    input_im = np.clip(np.power(input_im, r), 0, 1)
    return input_im, output


def add_scaling2(input_im, output, r_limits):

    min_scaling, max_scaling = r_limits
    scaling_factor = np.random.uniform(min_scaling, max_scaling)

    def crop_or_fill(image, shape):
        image = np.copy(image)
        for dimension in range(2):
            if image.shape[dimension] > shape[dimension]:
                # Crop
                if dimension == 0:
                    image = image[:shape[0], :]
                elif dimension == 1:
                    image = image[:, :shape[1], :]
            else:
                # Fill
                if dimension == 0:
                    new_image = np.zeros((shape[0], image.shape[1], shape[2]))
                    new_image[:image.shape[0], :, :] = image
                elif dimension == 1:
                    new_image = np.zeros((shape[0], shape[1], shape[2]))
                    new_image[:, :image.shape[1], :] = image
                image = new_image
        return image

    input_im = crop_or_fill(scipy.ndimage.zoom(input_im.astype(np.float32), [scaling_factor,scaling_factor,1], order=1), input_im.shape)
    output = crop_or_fill(scipy.ndimage.zoom(output.astype(np.float32), [scaling_factor, scaling_factor, 1], order=0), output.shape)

    return input_im, output


"""
performs intensity transform on the chunk, using gamma transform with random gamma-value
"""


def add_brightness_mult2(input_im, output, r_limits):
    # limits
    r_min, r_max = r_limits

    # randomly choose multiplication factor
    r = uniform(r_min, r_max)

    # apply aug
    input_im = np.clip(np.round(input_im * r), a_min=0, a_max=1)

    return input_im, output


def add_HEstain2(input_im, output):
    # RGB: float [0,1] -> uint8 [0,1] to use staintools
    #input_im = (np.round(255. * input_im.astype(np.float32))).astype(np.uint8)
    # input_im = input_im.astype(np.uint8)

    # standardize brightness (optional -> not really suitable for augmentation?
    # input_im = staintools.LuminosityStandardizer.standardize(input_im)

    # define augmentation algorithm -> should only do this the first time!
    if not 'augmentor' in globals():
        global augmentor
        # input_im = input_im[...,::-1]
        # augmentor = staintools.StainAugmentor(method='vahadane', sigma1=0.2, sigma2=0.2) # <- best, but slow
        augmentor = staintools.StainAugmentor(method='macenko', sigma1=0.1, sigma2=0.1)  # <- faster but worse

    # fit augmentor on current image
    augmentor.fit(input_im.astype(np.uint8))

    # extract augmented image
    input_im = augmentor.pop()

    #input_im = input_im.astype(np.float32) / 255.

    return input_im, output


def add_rotation2_ll(input_im, output):
    # randomly choose rotation angle: 0, +-90, +,180, +-270
    k = random_integers(0, high=3)  # -> 0 means no rotation

    # rotate
    input_im = np.rot90(input_im, k)
    output = np.rot90(output, k)

    return input_im, output


def add_hsv2(input_im, output, max_shift):
    # RGB: float [0,1] -> uint8 [0,1]
    #input_im = (np.round(255. * input_im.copy())).astype(np.uint8)

    # RGB -> HSV
    input_im = cv2.cvtColor(input_im.astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)

    ## augmentation, only on Hue and Saturation channel
    # hue
    input_im[..., 0] = np.mod(input_im[..., 0] + round(uniform(-max_shift, max_shift)), 180)

    # saturation
    input_im[..., 1] = np.clip(input_im[..., 1] + round(uniform(-max_shift, max_shift)), a_min=0, a_max=255)

    # input_im = (np.round(255*maxminscale(input_im.astype(np.float32)))).astype(np.uint8)

    # input_im = np.round(input_im).astype(np.uint8)
    input_im = input_im.astype(np.uint8)

    # HSV -> RGB
    input_im = cv2.cvtColor(input_im, cv2.COLOR_HSV2RGB)

    # need to normalize again after augmentation
    #input_im = (input_im.astype(np.float32) / 255)

    return input_im, output


def add_jpeg2(input_im, output, params):
    # augmentation parameters: (default: min_comp = 10, max_comp = 50, prob = 0.5)
    min_comp = params[0]
    max_comp = params[1]
    prob = params[2]

    # randomly select which compression
    comp_val = int(np.random.randint(min_comp, max_comp, 1))
    #print(comp_val)

    # randomly to compression or not
    comp_it = np.random.rand() <= prob

    col_dim = input_im.shape[-1]
    if not comp_it:
        return input_im, output
    elif col_dim == 1:
        mode = 'L'
        image = PIL.Image.fromarray(np.squeeze(input_im, axis=-1).astype(np.uint8), mode)
    elif col_dim == 3:
        mode = 'RGB'
        image = PIL.Image.fromarray(input_im.astype(np.uint8), mode)
    else:
        raise ValueError('Unsupported nr of channels in JPEGCompression transform')

    with BytesIO() as f:
        image.save(f, format='JPEG', quality=100-comp_val)
        f.seek(0)
        image_jpeg = PIL.Image.open(f)
        result = np.asarray(image_jpeg).astype(np.float32)
        result = result.copy()
    return result.reshape(input_im.shape), output




@nb.jit(nopython=True)
def sc_any(array):
    for x in array.flat:
        if x:
            return True
    return False


def maxminscale(tmp):
    if sc_any(tmp):
        tmp = tmp - np.amin(tmp)
        tmp = tmp / np.amax(tmp)
    return tmp



"""
aug: 		dict with what augmentation as key and what degree of augmentation as value
"""

def batch_gen2(file_list, batch_size, aug={}, shuffle_list=True, epochs=1, num_classes=9):
    cnt = 0
    batch = 0
    #for filename in file_list:
    file = h5py.File(file_list[0], 'r')
    input_shape = file['data'].shape
    output_shape = file['label'].shape

    # for each epoch, clear batch
    im = np.zeros((batch_size, input_shape[1], input_shape[2], 1), dtype=np.float32)
    gt = np.zeros((batch_size, output_shape[1], output_shape[2], output_shape[3]), dtype=np.uint8)

    for i in range(epochs):

        if shuffle_list:
            random.shuffle(file_list)

        for filename in file_list:
            file = h5py.File(filename, 'r')
            input_im = np.array(file['data'], dtype=np.float32)
            output = np.array(file['label'], dtype=np.uint8)
            file.close()

            input_im = np.squeeze(input_im, axis=0)
            output = np.squeeze(output, axis=0)

            # apply specified agumentation on both image stack and ground truth
            if 'stain' in aug:  # <- do this first
                input_im, output = add_HEstain2(input_im, output)

            if 'hsv' in aug:  # <- do this first
                input_im, output = add_hsv2(input_im, output, aug['hsv'])

            if 'gamma' in aug:
                input_im, output = add_gamma2(input_im, output, aug['gamma'])

            if 'mult' in aug:
                input_im, output = add_brightness_mult2(input_im, output, aug['mult'])

            if 'gauss' in aug:
                input_im, output = add_gaussian_blur2(input_im, output, aug['gauss'])

            if 'jpeg' in aug:
                input_im, output = add_jpeg2(input_im, output, aug['jpeg'])

            if 'flip' in aug:
                input_im, output = add_flip2(input_im, output)

            if 'rotate_ll' in aug:
                input_im, output = add_rotation2_ll(input_im, output)

            if 'shadow' in aug:
                input_im, output = add_gauss_shadow2(input_im, output, aug['shadow'])

            if 'affine' in aug:
                input_im, output = add_affine_transform2(input_im, output, aug['affine'])

            if 'rotate' in aug:  # -> do this last maybe?
                input_im, output = add_rotation2(input_im, output, aug['rotate'])

            if 'scale' in aug:
                input_im, output = add_scaling2(input_im, output, aug['scale'])

            if 'shift' in aug:
                input_im, output = add_shift2(input_im, output, aug['shift'])

            '''
            fig, ax = plt.subplots(1,2, figsize=(10,10))
            ax[0].imshow(input_im[..., 0])
            ax[1].imshow(output[..., 1], cmap="gray")
            plt.show()
            '''

            # normalize at the end
            #input_im = input_im.astype(np.float32) / 255.

            # insert augmented image and GT into batch
            im[batch] = input_im #np.expand_dims(input_im, axis=0)
            gt[batch] = output #np.expand_dims(output, axis=0)

            del input_im, output

            batch += 1
            if batch == batch_size:
                # reset and shuffle batch in the end
                batch = 0
                #names = np.array(range(batch_size))
                #shuffle(names)
                #im = im[names]
                #gt = gt[names]
                yield im, gt


def batch_length(file_list):
    length = len(file_list)
    # length = 0
    # for filename in file_list:
    # file = h5py.File(filename, 'r')
    # input_im = file['data']
    # length += 1
    # for pat in range(input_im.shape[0]):
    #	length = length + 1
    print('images in generator:', length)
    return length
