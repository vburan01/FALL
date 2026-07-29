# Dataset loading and spectral-band normalization for reproducibility.

from pathlib import Path

import numpy as np
import scipy.io


# Python slices are half-open, so this denotes inclusive rows 253--327 and
# columns 30--209, producing the reported 75 x 180 crop.
PAVIA_U_CROP = (253, 328, 30, 210)
PAVIA_U_CROP_SHAPE = (
    PAVIA_U_CROP[1] - PAVIA_U_CROP[0],
    PAVIA_U_CROP[3] - PAVIA_U_CROP[2],
)

# Each entry stores data file, GT file, MATLAB keys, and an optional crop.
DATASETS = {
    "salinasA": (
        "SalinasA.mat",
        "SalinasA_gt.mat",
        "salinasA",
        "salinasA_gt",
        None,
    ),
    "paviaU_crop": (
        "PaviaU.mat",
        "PaviaU_gt.mat",
        "paviaU",
        "paviaU_gt",
        PAVIA_U_CROP,
    ),
}

def _spectral_band_normalize(X):
    # Scale every wavelength band to unit L2 norm over the complete scene.

    X = np.asarray(X, dtype=float).copy()
    norms = np.linalg.norm(X, axis=0)
    positive = norms > 0
    X[:, positive] = X[:, positive] / norms[np.newaxis, positive]
    return X


def _apply_crop(data, gt, crop):
    # Apply a shared half-open spatial crop to the cube and ground truth.

    if crop is None:
        return data, gt

    row_start, row_stop, col_start, col_stop = crop
    rows, cols = gt.shape
    if not (0 <= row_start < row_stop <= rows):
        raise ValueError(f"PaviaU crop rows must satisfy 0 <= start < stop <= {rows}.")
    if not (0 <= col_start < col_stop <= cols):
        raise ValueError(f"PaviaU crop cols must satisfy 0 <= start < stop <= {cols}.")

    return data[row_start:row_stop, col_start:col_stop, :], gt[row_start:row_stop, col_start:col_stop]


def load_hsi_dataset(
    name="salinasA",
    dataset_dir="datasets",
    return_shape=False,
):
    # Load SalinasA or the fixed PaviaU crop used in the experiments.
    #
    # Spectral-band normalization is applied unconditionally because it is the
    # only preprocessing used by the reproducibility experiments.

    if name not in DATASETS:
        valid = ", ".join(sorted(DATASETS))
        raise ValueError(f"Unknown dataset '{name}'. Expected one of: {valid}.")

    data_file, gt_file, data_key, gt_key, crop = DATASETS[name]
    dataset_dir = Path(dataset_dir)

    # MATLAB files retain the original image cube and integer ground-truth map.
    mat_data = scipy.io.loadmat(dataset_dir / data_file)
    mat_gt = scipy.io.loadmat(dataset_dir / gt_file)
    data = mat_data[data_key]
    gt = mat_gt[gt_key]

    # Match the reported experiments: normalize each band over the complete
    # scene first, then extract the fixed PaviaU crop.
    scene_rows, scene_cols, bands = data.shape
    X = _spectral_band_normalize(data.reshape((scene_rows * scene_cols, bands)))
    data = X.reshape((scene_rows, scene_cols, bands))
    data, gt = _apply_crop(data, gt, crop)

    # Graph routines consume one row per pixel while visualizations retain the
    # corresponding two-dimensional spatial shape.
    rows, cols, bands = data.shape
    X = data.reshape((rows * cols, bands))
    gt = gt.reshape(-1)

    if return_shape:
        return X, gt, (rows, cols)
    return X, gt
