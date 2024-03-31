import h5py
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.keras.models import load_model
import cv2
from tqdm import tqdm
import tensorflow as tf
import os
from erik_code.post_process import post_process, post_process2, post_process3

from keras_tools.metaimage import *
from matplotlib.colors import Normalize


class PiecewiseNormalize(Normalize):
    def __init__(self, xvalues, cvalues):
        self.xvalues = xvalues
        self.cvalues = cvalues

        Normalize.__init__(self)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        if self.xvalues is not None:
            x, y = self.xvalues, self.cvalues
            return np.ma.masked_array(np.interp(value, x, y))
        else:
            return Normalize.__call__(self, value, clip)


def import_set(tmp, num=None, filter=False):
    f = h5py.File(datasets_path + 'dataset_' + name + '.h5', 'r')
    tmp = np.array(f[tmp])
    tmp = [tmp[i].decode("UTF-8") for i in range(len(tmp))]
    #shuffle(tmp)
    if filter:
        tmp = remove_copies(tmp)
    if num != None:
        tmp = tmp[:num]
    f.close()
    return tmp

def DSC(target, output):
    dice = 0
    #epsilon = 1e-10
    for object in range(1, output.shape[-1]):
        output1 = output[..., object]
        target1 = target[..., object]
        intersection = np.sum(output1 * target1)
        union = np.sum(output1 * output1) + np.sum(target1 * target1)
        dice += 2. * intersection / union
    dice /= (output.shape[-1] - 1)
    return dice

def DSC_simple(target, output):
    #epsilon = 1e-10
    intersection = np.sum(output * target)
    union = np.sum(output * output) + np.sum(target * target)
    dice = 2. * intersection / union
    return dice

def oneHoter(tmp, labels):
    res = np.zeros(tmp.shape + (labels, ))
    for l in range(labels):
        tmp2 = np.zeros_like(tmp)
        tmp2[tmp == l] = 1
        res[..., l] = tmp2
    return res


# set seed
np.random.seed(42)

# choose whether to run on GPU or not, and for which GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# set name for model
name = "13_07_multi_seg_4"
#name = "15_07_multi_seg_7_classes"
#name = "15_07_multi_seg_7_classes_spat_drop_0.2_256_deepest"
#name = "15_07_multi_seg_7_classes_spat_drop_0.2_deepest_bs_8"
name = "15_07_multi_seg_7_classes_spat_drop_0.2_deepest_bs_8_zoom_aug_adam"
name = "16_07_multi_seg_7_classes_spat_drop_0.2_deepest_bs_8_zoom_aug_adam"
name = "16_07_multi_seg_7_classes_spat_drop_0.2_deepest_bs_6_zoom_aug_adam_decay_stale_50_2000_epochs"

# number of classes
num_classes = 7

# paths
path = "/home/andrep/anesthesia/"
data_path = path + "data/anesthesia_seg_12_07/"
save_model_path = path + "output/models/"
history_path = path + "output/history/"
datasets_path = path + "output/datasets/"

gt_path = "/mnt/EncryptedData1/anesthesia/axillary/annotations/"

# pred threshold
th = 0.5

# load test_data_set
test_set = import_set('test')
val_set = import_set('val')
train_set = import_set('train')

# choose set
sets = test_set.copy()
sets = np.array(sets)

# load trained model
model = load_model(save_model_path + "model_" + name + '.h5', compile=False)

# make colormap for each class
vals = np.linspace(0, 1, num_classes)
np.random.shuffle(vals)
vals = plt.cm.jet(vals)
vals[0, :] = [0, 0, 0, 1]
cmap = plt.cm.colors.ListedColormap(vals)

# make colormap where one color represent each class consistently
norm = PiecewiseNormalize(vals, vals)

dsc_list = []
sens_list = []
prec_list = []

dsc_vals = np.zeros((len(sets), num_classes))-1

fig, ax = plt.subplots(1,6, figsize=(13,13))
titles = ["US", "Pred", "Filter1", "Filter2", "Filter3", "GT"]

#sets = ["/home/andrep/anesthesia/data/anesthesia_seg_12_07/002_0_US-2D_498.h5"]

counter = 0
for path in tqdm(np.array(sets)): #[(39+16):]): # (39+16) , # (39+16+11) <- weird artefact at bottom
    print(path)
    loc = path.split("/")[-1].split(".h5")[0]
    locs = loc.split("_")
    l1 = locs[0]
    l2 = locs[1]
    l3 = locs[2] + "_" + locs[3]
    loc = gt_path + l1 + "/" + l2 + "/" + l3 + "_gt" + ".mhd"
    data_object = MetaImage(loc)
    gt_orig = np.asarray(data_object.get_image())


    f = h5py.File(path, 'r')
    data = np.array(f['data'])
    gt = np.array(f['label'])
    f.close()

    print(data.shape)
    print(gt_orig.shape)

    factor = int(np.round(256 / gt_orig.shape[1] * gt_orig.shape[0]))

    pred = model.predict(data)

    data = data[:, :factor]
    gt = gt[:, :factor]
    pred = pred[:, :factor]

    print(data.shape)
    print(gt.shape)
    print(pred.shape)

    preds = np.zeros(data.shape[1:-1], dtype=np.float32)
    gts = np.zeros_like(preds)

    preds_oh = np.zeros_like(gt.copy(), dtype=np.float32)

    # binarize each class
    for c in range(1, gt.shape[-1]):
        tmp = np.squeeze(pred[..., c], axis=0)
        tmp[tmp <= th] = 0
        tmp[tmp > th] = c
        preds += tmp

        tmp2 = np.squeeze(gt[..., c], axis=0)*c
        gts += tmp2

        #preds_oh[..., c] = tmp.copy()/c

    #preds = post_process(preds) # <- post processing seemed to degrade DSC (?)
    preds_pp = post_process(preds.copy())
    preds_pp2 = post_process2(preds_pp.copy())
    preds_pp3 = post_process3(preds_pp2.copy())

    preds_tmp = preds_pp3.copy()
    print(preds.shape)
    print(preds_tmp.shape)
    print(preds_oh.shape)

    for c in range(1, gt.shape[-1]):
        tmp = np.zeros(preds.shape)
        tmp[preds_tmp == c] = 1
        print(tmp.shape)
        if len(np.unique(tmp)) > 1:
            preds_oh[..., c] = tmp.copy()

    # calculate DSC for each class and overall macro DSC (except background)
    dsc_tmp = []
    for c in range(gt.shape[-1]):
        gt_tmp = gt[..., c]
        pred_tmp = preds_oh[..., c]
        if len(np.unique(gt_tmp)) > 1 and len(np.unique(pred_tmp)) > 1:
            #dsc_tmp.append(DSC_simple(gt_tmp, pred_tmp))
            dsc_vals[counter, c] = DSC_simple(gt_tmp, pred_tmp)
        elif len(np.unique(gt_tmp)) > 1 and len(np.unique(pred_tmp)) == 1:
            #dsc_tmp.append(0)
            dsc_vals[counter, c] = 0

    print(dsc_vals[counter])
    ax[0].imshow(data[0, ..., 0], cmap="gray")
    ax[1].imshow(preds, cmap=cmap, vmin=0, vmax=num_classes-1)
    ax[2].imshow(preds_pp, cmap=cmap, vmin=0, vmax=num_classes-1)#,  #cmap=cmap)
    ax[3].imshow(preds_pp2,cmap=cmap, vmin=0, vmax=num_classes-1)
    ax[4].imshow(preds_pp3, cmap=cmap, vmin=0, vmax=num_classes-1)
    ax[5].imshow(gts, cmap=cmap, vmin=0, vmax=num_classes-1)

    for i in range(6):
        ax[i].set_title(titles[i])
        ax[i].set_xticks([])
        ax[i].set_yticks([])

    counter += 1

    plt.waitforbuttonpress(27)

#plt.show()

print(dsc_vals.shape)


# report macro average DSC for each class individually and overall
dsc_class = []
dsc_means = []
dsc_stds = []
for c in range(dsc_vals.shape[1]):
    tmp = []
    for i in range(dsc_vals.shape[0]):
        curr = dsc_vals[i, c]
        if curr != -1:
            tmp.append(curr)
    dsc_class.append(tmp)
    dsc_means.append(np.mean(tmp))
    dsc_stds.append(np.std(tmp))

    print(tmp)
    print()

print(dsc_means)
print(dsc_stds)

print('Average DSC: ')
print(np.mean(dsc_means[1:]))
