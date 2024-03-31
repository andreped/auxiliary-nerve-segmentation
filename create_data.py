from keras_tools.metaimage import *
import h5py
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2
from skimage import transform


def oneHoter(tmp, labels):
    res = np.zeros(tmp.shape + (labels, ))
    for l in range(labels):
        tmp2 = np.zeros_like(tmp)
        tmp2[tmp == l] = 1
        res[..., l] = tmp2
    return res


def resizer(data, out_dim, gt=True):
    orig_dim = data.shape
    scale = out_dim[0] / data.shape[1]
    if not gt:
        #data = transform.rescale(data, scale=scale, preserve_range=True, order=1, multichannel=False)  # This also transforms image to be between 0 and 1, therefore preserve_range=True
        data = cv2.resize(data.astype(np.uint8), (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR).astype(np.float32)
    else:
        #data = transform.rescale(data, scale=scale, preserve_range=True, order=0, multichannel=False)
        data = cv2.resize(data.astype(np.uint8), (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST).astype(np.float32)

    if data.shape[0] > orig_dim[0]:
        # cut image
        data = data[:orig_dim[0], :]
    elif data.shape[0] < orig_dim[0]:
        tmp = np.zeros(orig_dim, dtype=np.float32)
        tmp[:data.shape[0], :out_dim[0]] = data
        data = tmp[:out_dim[0], :out_dim[1]]
    return data


# folder to store generated data
name = "anesthesia_seg_12_07"

# paths
path = "/mnt/EncryptedData1/anesthesia/axillary/"
data_path = path + "ultrasound/"
gt_path = path + "annotations/"
end_path = "/home/andrep/anesthesia/data/" + name + "/"

if not os.path.exists(end_path):
    os.mkdir(end_path)

locs = []
for pat in os.listdir(gt_path):
    curr1 = gt_path + pat
    for scan in os.listdir(curr1):
        curr2 = curr1 + "/" + scan
        for img in os.listdir(curr2):
            loc_gt = curr2 + "/" + img
            if img.endswith(".mhd") and not "_gt" in loc_gt:
                locs.append(loc_gt)

# set seed
np.random.seed(42)

# number of classes
num_classes = 7

# make colormap for each class
vals = np.linspace(0, 1, 9)
np.random.shuffle(vals)
vals = plt.cm.jet(vals)
vals[0, :] = [0, 0, 0, 1]
cmap = plt.cm.colors.ListedColormap(vals)

# init
counts = [0]*num_classes
dims = []
vals = []

# generate data
for loc in tqdm(locs, "Scans: "):

    #print(loc)
    image = loc.split("/")[-1].split(".mhd")[0]
    scan = loc.split("/")[-2]
    id = loc.split("/")[-3]
    gt_loc = loc.split(".mhd")[0] + "_gt" + ".mhd"

    gt_object = MetaImage(gt_loc)
    gt = np.asarray(gt_object.get_image())

    if len(np.unique(gt)) > 1:

        data_object = MetaImage(loc)
        data = np.asarray(data_object.get_image())

        # resize image
        data = resizer(data, out_dim=(256, 256), gt=False)
        gt = resizer(gt, out_dim=(256, 256), gt=True)

        # flip if image is LEFT oriented to get same orientation for all images
        if gt_object.attributes['orientation'] == 'left':
            data = np.fliplr(data)
            gt = np.fliplr(gt)

        # merge class 3 & 7 -> 1, and re-label to get labels in range {0,1,...,6} -> 7 in total (only for convenience)
        gt[gt == 3] = 1
        gt[gt == 7] = 1
        gt[gt == 4] = 3
        gt[gt == 5] = 4
        gt[gt == 6] = 5
        gt[gt == 8] = 6

        vals.append(np.unique(gt))

        # normalize
        data = data.astype(np.float32) / 255

        # one-hot GT (should be nine classes in total, including background)
        GT = oneHoter(gt, num_classes)
        dims.append(list(gt.shape))

        data = np.expand_dims(data, axis=-1)
        data = np.expand_dims(data, axis=0)
        GT = np.expand_dims(GT, axis=0)

        # save as hdf5
        f = h5py.File(end_path + id + "_" + scan + "_" + image + '.h5', 'w')
        f.create_dataset('data', data=np.array(data), compression="gzip", compression_opts=4)
        f.create_dataset('label', data=np.array(GT), compression="gzip", compression_opts=4)
        f.close()









