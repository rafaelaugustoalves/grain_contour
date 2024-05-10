import h5py
import matplotlib.pyplot as plt


hdf_filename = r'D:\luciano_slice\new_seg_post_v1.h5'
with h5py.File(hdf_filename, 'r') as file:
    seg = file.get('data')[...]  # assumes data to be a numpy array
plt.imshow(seg, interpolation='None')
plt.show()
