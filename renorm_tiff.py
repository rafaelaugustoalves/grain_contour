import copy
import math
import os

import numpy
from numpy.typing import NDArray
import cv2
import h5py
import matplotlib.pyplot as plt
from matplotlib import cm, colors


def save_hdf(hdf_filename: str, original_file: str, data):
    with h5py.File(hdf_filename, 'w') as hdf_file:
        hdf_file.attrs.create(f'original_file', f'{original_file}')
        hdf_file.create_dataset('data', data=data)


def save_normalized_tiff(
    path: str,
    base_name: str,
    format: str,
    # new_max: int,
):
    # scan
    filename = os.path.join(path, f'{base_name}{format}')
    gray_image = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
    print(numpy.min(gray_image), numpy.max(gray_image))
    print(gray_image.dtype)
    print(gray_image.shape)

    hdf_filename = os.path.join(path, 'scan.h5')
    save_hdf(hdf_filename, f'{base_name}{format}', gray_image)

    min_val = numpy.min(gray_image)
    max_val = numpy.max(gray_image)
    new_max = max_val

    aux = gray_image.astype(dtype=float)
    print(f'old Aux: {numpy.max(aux)}, {numpy.min(aux)}')

    out_range_mask = aux > new_max
    aux[out_range_mask] = new_max
    val_range = new_max - min_val

    aux -= min_val
    aux = numpy.divide(aux, val_range)
    print(f'Aux: {numpy.max(aux)}, {numpy.min(aux)}')
    aux *= 64000
    aux = aux.astype(dtype='uint16')
    print(numpy.percentile(aux, [0.1, 0.2, 0.3, 0.4, 0.5]))

    out_filename = os.path.join(path, f'normalized-{new_max}_{base_name}.png')
    cv2.imwrite(out_filename, aux)

    hdf_filename = os.path.join(path, f'normalized-{new_max}_{base_name}.h5')
    save_hdf(hdf_filename, f'{base_name}{format}', aux)

    # seg
    # data = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
    # gray_image = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)


if __name__ == '__main__':
    path = r'D:\luciano_slice'
    base_name = '2_122.45.XZ959_cropped'
    save_normalized_tiff(path, base_name, '.tif',)


    base_name = '3_BINARIZADA_122.45.XZ959_cropped.tif'
    filename = os.path.join(path, f'{base_name}')
    data = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
    gray_image = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
    print(numpy.unique(gray_image))

    gray_image[gray_image == 255] = 1
    out_filename = os.path.join(path, '3_sed.tif')
    cv2.imwrite(out_filename, gray_image)

    hdf_filename = os.path.join(path, f'3_seg.h5')
    save_hdf(hdf_filename, f'{base_name}.tif', gray_image)