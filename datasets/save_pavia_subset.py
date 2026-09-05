#Creates the Pavia University subset used in Fermat Active Learning. Note that
#this script requires PaviaU.mat and PaviaU_gt.mat to be in the same folder 
# as this script. The output MAT files will be saved in the same folder.

from pathlib import Path

import scipy.io

from preprocessing import load_hsi_dataset


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_FILENAME = "PaviaU.mat"
GT_FILENAME = "PaviaU_gt.mat"


def main():
    for filename in (DATA_FILENAME, GT_FILENAME):
        input_path = SCRIPT_DIR / filename
        if not input_path.is_file():
            raise FileNotFoundError(
                f"Expected {filename} in the same folder as this script: "
                f"{SCRIPT_DIR}"
            )

    # load_hsi_dataset normalizes each spectral band over the full scene and
    # then extracts rows 253:328 and columns 30:210.
    X, gt, (rows, cols) = load_hsi_dataset(
        name="paviaU_crop",
        dataset_dir=SCRIPT_DIR,
        return_shape=True,
    )

    cube = X.reshape(rows, cols, X.shape[1])
    ground_truth = gt.reshape(rows, cols)

    data_output = SCRIPT_DIR / "PaviaU_reduced.mat"
    gt_output = SCRIPT_DIR / "PaviaU_reduced_gt.mat"

    scipy.io.savemat(
        data_output,
        {"paviaU": cube},
        do_compression=True,
    )
    scipy.io.savemat(
        gt_output,
        {"paviaU_gt": ground_truth},
        do_compression=True,
    )

    print(f"Saved hyperspectral cube: {data_output}")
    print(f"Cube shape:               {cube.shape}")
    print(f"Saved ground truth:       {gt_output}")
    print(f"Ground-truth shape:       {ground_truth.shape}")


if __name__ == "__main__":
    main()
