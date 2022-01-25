import os
import numpy as np
from PIL import Image
# should return three datasets: amazon, dlsr, webcam

def office_31_subset(dataset):
    xs = []
    ys = []
    dataset_path = os.path.join('office-31', dataset, 'images')
    for category in os.listdir(dataset_path):
        category_path = os.path.join('office-31', dataset, 'images', category)

        category_index = os.listdir(dataset_path).index(category)
        for image_name in os.listdir(category_path):
            standard_size = (56, 56)
            image = np.asarray(Image.open(os.path.join('office-31', dataset, 'images', category, image_name)).resize(standard_size))
            xs.append(image)
            ys.append(category_index)

    xs = np.asarray(xs)
    ys = np.asarray(ys)

    print("{} of office 31 has xs of shape {} and ys of shape {}".format(dataset, xs.shape, ys.shape))

    return xs, ys
#office_31_subset('amazon')
#office_31_subset('dslr')
#office_31_subset('webcam')
