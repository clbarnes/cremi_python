#!/usr/bin/env python
import os
import warnings

import numpy as np
import skimage.io as io
from skimage.viewer import CollectionViewer

import cremi.evaluation as evaluation
import cremi.io.CremiFile as CremiFile


def abspath(local_path):
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), local_path)


if __name__ == "__main__":
    img = [io.imread(abspath('example.png'))]
    
    for w in (0, 2, 4, 10):
        target = np.copy(img[0])[...,np.newaxis]
        evaluation.create_border_mask(img[0][...,np.newaxis],target,w,105,axis=2)
        img.append(target[...,0])

    try:
        v = CollectionViewer(img)
        v.show()
    except AttributeError as e:
        if 'Qt' in str(e):
            warnings.warn('Qt-related error: CollectionViewer not working')
        else:
            raise

    cfIn  = CremiFile(abspath('example.h5'), 'r')
    cfOut = CremiFile(abspath('output.h5'), 'w')

    evaluation.create_and_write_masked_neuron_ids(cfIn, cfOut, 3, 240, overwrite=True)
