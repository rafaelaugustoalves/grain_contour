import copy

import h5py
from scipy import ndimage
import matplotlib.pyplot as plt
import numpy
from numpy.typing import NDArray

import dijkstra3d


def read_hdf_dataset(
    filename: str,
    data_name: str,
) -> NDArray:
    with h5py.File(filename, 'r') as file:
        file_data = file.get(data_name)[...]  # assumes data to be a numpy array
    return file_data


def make_binary(data: numpy.ndarray, foreground_label: int) -> numpy.ndarray:
    foreground_mask = data == foreground_label
    binary_data = numpy.zeros_like(data, dtype=bool)
    binary_data[foreground_mask] = True
    return binary_data


def sample_random(surface: numpy.ndarray, max_num_samples: int) -> numpy.ndarray:
    # make sure we can select the required amount
    assert numpy.count_nonzero(surface) > max_num_samples

    idx_nonzero = numpy.argwhere(surface)
    random_indices = set()
    while len(random_indices) <= max_num_samples:
        num = numpy.random.randint(0, idx_nonzero.shape[0])
        ind = idx_nonzero[num, :]
        random_indices.add(tuple(ind))

    num = len(random_indices)
    indices = numpy.array(list(random_indices), dtype=int).reshape((num, 2))

    return indices


def erode_2d(data: numpy.ndarray, num_erosion: int) -> numpy.ndarray:
    if len(numpy.unique(data)) != 2:
        raise ValueError(
            'Erosion should be performed on array with only background '
            '(zero/False) and one label (1/True)'
        )

    # connectivity: 1 is faces, 2 is sides, 3 is vertices
    struct = ndimage.generate_binary_structure(rank=2, connectivity=1)
    eroded_data = copy.deepcopy(data)
    for counter in range(num_erosion):
        eroded_data = ndimage.binary_erosion(eroded_data, struct)
        print(f'eroded {counter + 1} times')

    return eroded_data


def find_surface(binary_data: numpy.ndarray) -> numpy.ndarray:
    eroded_data = erode_2d(binary_data, 1)
    surface = binary_data ^ eroded_data
    return surface


def save_hdf(hdf_filename: str, original_file: str, data):
    with h5py.File(hdf_filename, 'w') as hdf_file:
        hdf_file.attrs.create(f'original_file', f'{original_file}')
        hdf_file.create_dataset('data', data=data)


def main():
    seg_filename = r'D:\luciano_slice\3_seg.h5'
    segmentation = read_hdf_dataset(seg_filename, 'data')
    print(segmentation.dtype)
    print(segmentation.shape)

    scan_filename = r'D:\luciano_slice\scan.h5'
    scan = read_hdf_dataset(scan_filename, 'data')

    weight_matrix = scan.astype(dtype='uint32')
    print(weight_matrix.dtype)
    print(weight_matrix.shape)

    pore_mask = segmentation == 0
    pore_weight = 1
    weight_matrix[pore_mask] = pore_weight

    solid_mask = segmentation == 2
    solid_weight = 4294967295 - 10  # uint32, minus 10 to be safe
    weight_matrix[solid_mask] = solid_weight

    print('start getting k_closest')
    numpy.random.seed(0)

    num_sources = 500 #int(1700 * 2200 / 10)
    binary_array_1 = make_binary(segmentation, 0)
    surface_1 = find_surface(binary_array_1)
    sources = sample_random(surface_1, num_sources)
    targets = numpy.roll(sources, shift=1, axis=0)

    new_seg = copy.deepcopy(segmentation)
    paths = []
    costs = []
    counter1 = 1
    for source, target in zip(sources, targets):
        voxels_path = dijkstra3d.dijkstra(
            weight_matrix, source, target, connectivity=8, bidirectional=True)

        voxels_cost = weight_matrix[voxels_path[:, 0], voxels_path[:, 1]]
        if solid_weight in voxels_cost:
            print(f'Got Solid on path {counter1}')
            continue
        paths.append(voxels_path)
        costs.append(voxels_cost)

        new_seg[voxels_path[:, 0], voxels_path[:, 1]] = 7

        print(f'Done {counter1} out of {num_sources} source/target paths ({counter1/num_sources * 100 :.4f}%)')
        counter1 += 1

        hdf_filename = r'D:\luciano_slice\new_seg_raw_v1.h5'
        save_hdf(hdf_filename, 'new_seg', new_seg)

        new_seg_post = copy.deepcopy(new_seg)
        new_seg_post[pore_mask] = 0
        hdf_filename = r'D:\luciano_slice\new_seg_post_v1.h5'
        save_hdf(hdf_filename, 'new_seg_post', new_seg_post)

        plt.imshow(new_seg, interpolation='None')
        plt.show()

        plt.imshow(new_seg_post, interpolation='None')
        plt.show()


if __name__ == '__main__':
    main()