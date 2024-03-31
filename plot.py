import matplotlib.pyplot as plt
import numpy as np
from skimage.segmentation import find_boundaries
from skimage import transform
import PIL


def create_result_image(example, predicted_output, true_positive, false_positive, false_negative, filename=None):
    plt.figure(figsize=(10, 10))
    plt.axis('off')
    result_image = np.zeros((example['original_image'].shape[0], example['original_image'].shape[1], 3))
    for i in range(3):
        result_image[:, :, i] = example['original_image']
    alpha = 0.5
    colors = np.array([
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0],
        [1, 0, 1],
        [0, 1, 1],
    ])
    transformed_output = transform.rescale(predicted_output, 1.0/example['scale'], preserve_range=True)
    transformed_output = transformed_output[:result_image.shape[0], :]
    for i in range(1, 6):
        transformed_segmentation = transform.rescale(example['segmentation'][:, :, i], 1.0/example['scale'], preserve_range=True)
        transformed_segmentation = transformed_segmentation[:result_image.shape[0], :]
        print(transformed_output.shape)
        print(result_image.shape)
        result_image[transformed_output == i, :] = (1.0 - alpha)*result_image[transformed_output == i, :] + alpha*colors[i - 1, :]
        result_image[find_boundaries(transformed_segmentation > 0, mode='outer'), :] = colors[i - 1, :]



    if filename is None:
        plt.imshow(result_image, cmap='gray')
        plt.title('True positives:' + ' '.join(str(e) for e in true_positive) + '\n' +
                  'False positives: ' + ' '.join(str(e) for e in false_positive) + '\n' +
                  'False negatives: ' + ' '.join(str(e) for e in false_negative))
        plt.show()
    else:
        #plt.savefig(filename, bbox_inches='tight')
        pillow_image = PIL.Image.fromarray((result_image*255).astype(np.uint8))
        resize_scale = 512/pillow_image.width
        new_height = int(round(pillow_image.height*resize_scale))
        pillow_image = pillow_image.resize((512, new_height))
        pillow_image = pillow_image.crop((0, 0, 512, 256))
        pillow_image.save(filename)
    plt.close()