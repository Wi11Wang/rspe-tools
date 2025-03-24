import numpy as np
import dask.array as da
from skimage.util import view_as_blocks
from numba import njit

@njit
def _get_bboxes_helper(_mask, _max_label, _offset):
    """
    Compute bounding boxes for labelled regions in a 3D mask array.

    :param ndarray _mask: 3D array where each voxel is labelled with an integer.
    :param int _max_label: Maximum number of labels.
    :param list _offset: Offset to adjust the bounding box coordinates.
    :return: Array of bounding boxes with shape (1, _max_label, 6), each as [x_min, y_min, z_min, x_max, y_max, z_max].
    :rtype: ndarray
    """
    bboxes = np.full((1, _max_label, 6), 32767, dtype=np.int16)
    bboxes[..., 3:] = -1
    for x in range(_mask.shape[0]):
        for y in range(_mask.shape[1]):
            for z in range(_mask.shape[2]):
                label = _mask[x, y, z]
                if label < _max_label:  # 0 is background
                    x_min, y_min, z_min, x_max, y_max, z_max = bboxes[0, label]
                    real_x = x + _offset[0]
                    real_y = y + _offset[1]
                    real_z = z + _offset[2]
                    bboxes[0, label, 0] = min(x_min, real_x)
                    bboxes[0, label, 1] = min(y_min, real_y)
                    bboxes[0, label, 2] = min(z_min, real_z)
                    bboxes[0, label, 3] = max(x_max, real_x)
                    bboxes[0, label, 4] = max(y_max, real_y)
                    bboxes[0, label, 5] = max(z_max, real_z)
    return bboxes

def get_bboxes_helper(mask, max_label, block_info=None):
    """
    Compute bounding boxes for a 3D mask using Dask block metadata.

    :param ndarray mask: 3D array where each voxel is labelled with an integer.
    :param int max_label: Maximum label number.
    :param dict block_info: Dask block information to compute offsets.
    :return: Array of bounding boxes for the provided mask block.
    :rtype: ndarray
    """
    offset = [loc[0] for loc in block_info[0]['array-location']]
    return _get_bboxes_helper(mask, _max_label=max_label, _offset=offset)

def get_bboxes(arr):
    """
    Aggregate bounding boxes from a Dask array across all blocks.

    :param ndarray arr: 3D array with labelled regions.
    :return: Aggregated array of bounding boxes, each as [x_min, y_min, z_min, x_max, y_max, z_max].
    :rtype: ndarray
    """
    MAX_LABEL = 10_000
    blocked_res = da.map_blocks(get_bboxes_helper, arr, max_label=MAX_LABEL, chunks=(1, MAX_LABEL, 6), dtype=np.int16).compute()
    res_reshaped = view_as_blocks(blocked_res, block_shape=(1, MAX_LABEL, 6)).reshape(-1, MAX_LABEL, 6)
    res_arr = np.empty((MAX_LABEL, 6), dtype=np.int16)
    res_arr[:, :3] = np.min(res_reshaped[:, :, :3], axis=0)
    res_arr[:, 3:] = np.max(res_reshaped[:, :, 3:], axis=0)
    return res_arr