# Description: Preprocess the BraTS2021 dataset into numpy arrays
# adpated from https://github.com/AntanasKascenas/DenoisingAE/tree/master

import torch
import random
from pathlib import Path
import numpy as np
try:
    import nibabel as nib
except Exception:
    nib = None
from tqdm import tqdm
try:
    import nrrd
except Exception:
    nrrd = None
import torch.nn.functional as F


def _load_nii(path: Path):
    if nib is not None:
        return nib.load(str(path)).get_fdata()

    import gzip
    import struct

    path_str = str(path)
    opener = gzip.open if path_str.endswith(".gz") else open
    with opener(path_str, "rb") as f:
        hdr = f.read(348)
        if len(hdr) != 348:
            raise ValueError(f"Invalid NIfTI header: {path_str}")

        endian = "<" if struct.unpack("<i", hdr[0:4])[0] == 348 else ">"
        sizeof_hdr = struct.unpack(endian + "i", hdr[0:4])[0]
        if sizeof_hdr != 348:
            raise ValueError(f"Invalid NIfTI header size: {path_str}")

        dim = struct.unpack(endian + "8h", hdr[40:56])
        shape = tuple(int(x) for x in dim[1:4])

        datatype = struct.unpack(endian + "h", hdr[70:72])[0]
        vox_offset = struct.unpack(endian + "f", hdr[108:112])[0]
        if vox_offset < 348:
            vox_offset = 348.0

        dt_map = {
            2: np.uint8,
            4: np.int16,
            8: np.int32,
            16: np.float32,
            64: np.float64,
            256: np.int8,
            512: np.uint16,
            768: np.uint32,
            1024: np.int64,
            1280: np.uint64,
        }
        if datatype not in dt_map:
            raise ValueError(f"Unsupported NIfTI datatype {datatype}: {path_str}")
        dtype = dt_map[datatype]

        f.seek(int(vox_offset))
        nvox = int(shape[0] * shape[1] * shape[2])
        data = f.read(nvox * np.dtype(dtype).itemsize)
        if len(data) != nvox * np.dtype(dtype).itemsize:
            raise ValueError(f"Truncated NIfTI data: {path_str}")

        arr = np.frombuffer(data, dtype=dtype).copy().reshape(shape, order="F")
        return arr.astype(np.float64, copy=False)


def _bratsped_required_files(case_dir: Path):
    stem = case_dir.name
    return [
        case_dir / f"{stem}-t2f.nii.gz",
        case_dir / f"{stem}-t1n.nii.gz",
        case_dir / f"{stem}-t1c.nii.gz",
        case_dir / f"{stem}-t2w.nii.gz",
        case_dir / f"{stem}-seg.nii.gz",
    ]


def _bratsped_missing_files(case_dir: Path):
    missing = []
    for p in _bratsped_required_files(case_dir):
        if not p.exists():
            missing.append(p)
    return missing


def normalise_percentile(volume):
    """
    Normalise the intensity values in each modality by scaling by 99 percentile foreground (nonzero) value.
    """
    for mdl in range(volume.shape[1]):
        v_ = volume[:, mdl, :, :].reshape(-1)
        v_ = v_[v_ > 0]  # Use only the brain foreground to calculate the quantile
        p_99 = torch.quantile(v_, 0.99)
        volume[:, mdl, :, :] /= p_99
    return volume

def center_crop(volume, target_shape):
    h, w, _ = volume.shape
    th, tw = target_shape, target_shape
    x1 = int(round((w - tw) / 2.))
    y1 = int(round((h - th) / 2.))
    cropped_volume = volume[y1:y1+th, x1:x1+tw, :]
    return cropped_volume
    

def process_patient(name, path, target_path, mod, first=-1, last=-1, downsample=False):
    
    if name == 'brats':
        flair = _load_nii(path / f"{path.name}_flair.nii.gz")
        t1 = _load_nii(path / f"{path.name}_t1.nii.gz")
        t1ce = _load_nii(path / f"{path.name}_t1ce.nii.gz")
        t2 = _load_nii(path / f"{path.name}_t2.nii.gz")
        labels = _load_nii(path / f"{path.name}_seg.nii.gz")
    elif name == 'bratsped':
        flair = _load_nii(path / f"{path.name}-t2f.nii.gz")
        t1 = _load_nii(path / f"{path.name}-t1n.nii.gz")
        t1ce = _load_nii(path / f"{path.name}-t1c.nii.gz")
        t2 = _load_nii(path / f"{path.name}-t2w.nii.gz")
        labels = _load_nii(path / f"{path.name}-seg.nii.gz")
    elif name == "atlas":
        t1 = _load_nii(path / f"{path.name}_T1w.nii.gz")
        labels = _load_nii(path / f"{path.name}_mask.nii.gz")
    elif name == 'mmbrain':
        if nrrd is None:
            raise ModuleNotFoundError("nrrd is required for dataset mmbrain")
        seed = random.randint(1, 5)
        flair = center_crop(nrrd.read(path / f"TrialSeed{seed}_FLAIR.nrrd")[0], 240).astype(np.float64)
        t1 = center_crop(nrrd.read(path / f"TrialSeed{seed}_T1.nrrd")[0], 240).astype(np.float64)
        t1ce = center_crop(nrrd.read(path / f"TrialSeed{seed}_T1Gad.nrrd")[0], 240).astype(np.float64)
        t2 = center_crop(nrrd.read(path / f"TrialSeed{seed}_T2.nrrd")[0], 240).astype(np.float64)
        labels = center_crop(nrrd.read(path / f"TrialSeed{seed}_discrete_truth.nrrd")[0], 240).astype(np.float64)
    elif name == "mslub":
        flair = np.moveaxis(_load_nii(path / f"{path.name}_FLAIR.nii.gz"), 0, -1)
        labels = np.moveaxis(_load_nii(path / f"{path.name}_consensus_gt.nii.gz"), 0, -1)
    else:
        raise ValueError(f"Dataset {name} not supported.")

    assert mod in ["all", "flair", "t1", "t1ce", "t2"]

    # volume shape: [1, 1, h, w, slices]
    if mod == "all":
        volume = torch.stack([torch.from_numpy(x) for x in [flair, t1, t1ce, t2]], dim=0).unsqueeze(dim=0)
    elif mod == "flair":
        volume = torch.stack([torch.from_numpy(x) for x in [flair]], dim=0).unsqueeze(dim=0)
    elif mod == "t1":
        volume = torch.stack([torch.from_numpy(x) for x in [t1]], dim=0).unsqueeze(dim=0)
    elif mod == "t1ce":
        volume = torch.stack([torch.from_numpy(x) for x in [t1ce]], dim=0).unsqueeze(dim=0)
    elif mod == "t2":
        volume = torch.stack([torch.from_numpy(x) for x in [t2]], dim=0).unsqueeze(dim=0)

    # exclude first n and last m slices
    # 1 4 240 240 155; 240 240 155
    
    if first > 0 and last > 0:
        volume = volume[:, :, :, :, first:-last]
        labels = labels[:, :, first:-last]
    elif first > 0 and last < 0:
        volume = volume[:, :, :, :, first:]
        labels = labels[:, :, first:]
    elif first < 0 and last > 0:
        volume = volume[:, :, :, :, :-last]
        labels = labels[:, :, :-last]
        
    # 1 1 240 240 155
    if name == 'brats' or name == 'bratsped' or name == 'mslub' or name == 'atlas':
        labels = torch.from_numpy(labels > 0.5).float().unsqueeze(dim=0).unsqueeze(dim=0)
    elif name == 'mmbrain':
        labels = torch.where(torch.from_numpy(labels)==5, 1, 0).float().unsqueeze(dim=0).unsqueeze(dim=0)

    patient_dir = target_path / f"patient_{path.name}"
    patient_dir.mkdir(parents=True, exist_ok=True)

    volume = normalise_percentile(volume)

    sum_dim2 = (volume[0].mean(dim=0).sum(axis=0).sum(axis=0) > 0.5).int()
    fs_dim2 = sum_dim2.argmax()
    ls_dim2 = volume[0].mean(dim=0).shape[2] - sum_dim2.flip(dims=[0]).argmax()

    print(f"Patient {path.name} has {fs_dim2} to {ls_dim2} slices with brain tissue.", flush=True)
    
    for slice_idx in range(fs_dim2, ls_dim2):
        if downsample:
            if name == 'brats' or name == 'bratsped' or name == 'atlas':
                low_res_x = F.interpolate(volume[:, :, :, :, slice_idx], mode="bilinear", size=(128, 128))
                low_res_y = F.interpolate(labels[:, :, :, :, slice_idx], mode="bilinear", size=(128, 128))
            elif name == 'mslub':
                low_res_x = F.interpolate(volume[:, :, :, :, slice_idx], mode="bilinear", size=(256, 256))
                low_res_y = F.interpolate(labels[:, :, :, :, slice_idx], mode="bilinear", size=(256, 256))
        else:
            low_res_x = volume[:, :, :, :, slice_idx]
            low_res_y = labels[:, :, :, :, slice_idx]
        np.savez_compressed(patient_dir / f"slice_{slice_idx}", x=low_res_x, y=low_res_y)


def preprocess(
    name: str,
    datapath: Path,
    mod: str,
    first=-1,
    last=-1,
    shape=128,
    downsample=True,
    output_dir: Path = None,
    case: str = "",
):

    if name == "bratsped":
        all_case_dirs = [p for p in datapath.iterdir() if p.is_dir()]
        all_case_dirs.sort(key=lambda p: p.name)
        all_imgs = [p for p in all_case_dirs if not _bratsped_missing_files(p)]
    elif str(case).strip():
        all_imgs = [datapath / str(case).strip()]
    else:
        all_imgs = sorted(list((datapath).iterdir()))

    if str(case).strip():
        all_imgs = [datapath / str(case).strip()]

    sub_dir = f"preprocessed_data_{mod}_{first}{last}_{shape}"
    base_dir = output_dir if output_dir is not None else datapath.parent
    splits_path = base_dir / sub_dir / "data_splits"

    if str(case).strip():
        target_path = base_dir / sub_dir / "npy_all"
        process_patient(name, all_imgs[0], target_path, mod, first, last, downsample=downsample)
        return

    if not splits_path.exists():

        indices = list(range(len(all_imgs)))
        random.seed(10)
        random.shuffle(indices)

        if name == 'brats' or name == 'bratsped':
            n_train = int(len(indices) * 0.75)
            n_val = int(len(indices) * 0.05)
            n_test = len(indices) - n_train - n_val
        elif name == 'atlas':
            n_train = int(len(indices) * 0.75)
            n_val = int(len(indices) * 0.05)
            n_test = len(indices) - n_train - n_val
        elif name == 'mmbrain':
            n_train = 0
            n_val = 5
            n_test = len(indices) - n_train - n_val
        elif name == 'MSLUB':
            n_train = 20
            n_val = 5
            n_test = 5

        split_indices = {}
        split_indices["train"] = indices[:n_train]
        split_indices["val"] = indices[n_train:n_train + n_val]
        split_indices["test"] = indices[n_train + n_val:]

        for split in ["train", "val", "test"]:
            (splits_path / split).mkdir(parents=True, exist_ok=True)
            with open(splits_path / split / "scans.csv", "w") as f:
                f.write("\n".join([all_imgs[idx].name for idx in split_indices[split]]))

    for split in ["train", "val", "test"]:
        paths = [datapath / x.strip() for x in open(splits_path / split / "scans.csv").readlines()]
        if name == "bratsped":
            paths = [p for p in paths if p.exists() and not _bratsped_missing_files(p)]
        else:
            paths = [p for p in paths if p.exists()]

        print(f"Patients in {split}]: {len(paths)}")

        for source_path in tqdm(paths):
            target_path = base_dir / sub_dir / f"npy_{split}"
            process_patient(name, source_path, target_path, mod, first, last, downsample=downsample)


if __name__ == "__main__":
   
    import argparse
    import shutil

    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--source", default='/data/amciilab/yiming/DATA/BraTS21_training/BraTS21', type=str, help="path to Brats2021 Data directory")
    parser.add_argument("--name", default='brats', type=str, help="dataset name")
    parser.add_argument("-m", "--mod", default='all', type=str, help="modelity to preprocess")
     
    # parser.add_argument("-s", "--source", default='/data/amciilab/yiming/DATA/MSLUB/data', type=str, help="path to data directory")
    # parser.add_argument("--name", default='MSLUB', type=str, help="dataset name")
    # parser.add_argument("--mod", default='all', type=str, help="modelity to preprocess")
    
    # parser.add_argument("-s", "--source", default='/data/amciilab/yiming/DATA/ATLAS/raw2', type=str, help="path to data directory")
    # parser.add_argument("--name", default='atlas', type=str, help="dataset name")
    # parser.add_argument("--mod", default='t1', type=str, help="modelity to preprocess")
     
     
    parser.add_argument("--first", default=0, 
                        type=int, help="skip first n slices")
    parser.add_argument("--last", default=0,
                        type=int, help="skip last n slices")
    parser.add_argument("--output", default="",
                        type=str, help="output directory")
    parser.add_argument("--case", default="",
                        type=str, help="process a single case folder name")
    parser.add_argument(
        "--clean-incomplete",
        action="store_true",
        help="delete BraTS-PED case folders missing required files",
    )
    
    args = parser.parse_args()

    datapath = Path(args.source)
    if args.clean_incomplete:
        if args.name != "bratsped":
            raise ValueError("--clean-incomplete is only supported for --name bratsped")

        all_case_dirs = [p for p in datapath.iterdir() if p.is_dir()]
        all_case_dirs.sort(key=lambda p: p.name)

        deleted = 0
        for case_dir in all_case_dirs:
            missing = _bratsped_missing_files(case_dir)
            if not missing:
                continue
            print(f"Deleting incomplete case: {case_dir} (missing {', '.join([m.name for m in missing])})", flush=True)
            shutil.rmtree(case_dir)
            deleted += 1

        print(f"Deleted {deleted} incomplete BraTS-PED cases.", flush=True)
   
    out_dir = Path(args.output) if str(args.output).strip() else None
    preprocess(
        args.name,
        datapath,
        args.mod,
        args.first,
        args.last,
        downsample=True,
        output_dir=out_dir,
        case=args.case,
    )
