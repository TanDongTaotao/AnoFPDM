import os
from tqdm import tqdm
import argparse
from pathlib import Path
import gzip
import shutil
import concurrent.futures
from typing import Iterable, List, Tuple

def _compute_output_path(nii_path: Path, root_path: Path, output_path: Path, brats_ped_flatten: bool) -> Path:
    relative_path = nii_path.relative_to(root_path)
    if not brats_ped_flatten:
        return output_path / relative_path.with_suffix(".nii.gz")

    parts = relative_path.parts
    if len(parts) >= 3:
        case_dir = parts[0]
        modality_dir = parts[1]
        if modality_dir.endswith(".nii"):
            return output_path / case_dir / f"{modality_dir}.gz"

    return output_path / relative_path.with_suffix(".nii.gz")


def _iter_nii_paths(root_path: Path) -> Iterable[Path]:
    for subdir, _, files in os.walk(root_path):
        for file in files:
            if file.endswith(".nii"):
                yield Path(subdir) / file


def _convert_one(task: Tuple[str, str, int, int]) -> Tuple[str, str]:
    src, dst, compresslevel, buffer_size = task
    src_path = Path(src)
    dst_path = Path(dst)

    if dst_path.exists():
        return ("skip", dst)

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with open(src_path, "rb") as f_in:
        with gzip.open(dst_path, "wb", compresslevel=compresslevel) as f_out:
            shutil.copyfileobj(f_in, f_out, length=buffer_size)
    return ("ok", dst)


def convert_nii_to_niigz(
    root_dir,
    output_dir,
    brats_ped_flatten: bool = False,
    workers: int = 0,
    compresslevel: int = 1,
    buffer_size: int = 1024 * 1024,
):
    """
    Converts all .nii files in subdirectories of root_dir to .nii.gz format,
    saving them to a new directory while preserving the folder structure.

    Args:
        root_dir (str): The path to the root directory containing patient folders.
        output_dir (str): The path to the directory where converted files will be saved.
    """
    root_path = Path(root_dir)
    output_path = Path(output_dir)

    tasks: List[Tuple[str, str, int, int]] = []
    for nii_path in _iter_nii_paths(root_path):
        nii_gz_path = _compute_output_path(
            nii_path=nii_path,
            root_path=root_path,
            output_path=output_path,
            brats_ped_flatten=brats_ped_flatten,
        )
        tasks.append((str(nii_path), str(nii_gz_path), int(compresslevel), int(buffer_size)))

    if not tasks:
        print("No .nii files found.")
        return

    if workers is None or workers <= 0:
        workers = os.cpu_count() or 1
    workers = max(1, int(workers))

    ok = 0
    skipped = 0
    errors = 0

    if workers == 1:
        for task in tqdm(tasks, total=len(tasks), desc="Converting"):
            try:
                status, _ = _convert_one(task)
                if status == "ok":
                    ok += 1
                else:
                    skipped += 1
            except Exception:
                errors += 1
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(_convert_one, task) for task in tasks]
            for fut in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Converting"):
                try:
                    status, _ = fut.result()
                    if status == "ok":
                        ok += 1
                    else:
                        skipped += 1
                except Exception:
                    errors += 1

    print(f"Done. ok={ok}, skipped={skipped}, errors={errors}, workers={workers}, compresslevel={compresslevel}, buffer_size={buffer_size}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert .nii files to .nii.gz format.')
    parser.add_argument('-d', '--directory', type=str, required=True, 
                        help='Path to the root directory containing the subfolders with .nii files.')
    parser.add_argument('-o', '--output', type=str, required=True,
                        help='Path to the output directory to save .nii.gz files.')
    parser.add_argument(
        '--brats-ped-flatten',
        action='store_true',
        help='Flatten BraTS-PED modality subfolders named like "*.nii" into per-case files.',
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=0,
        help='Number of parallel worker processes. 0 means auto.',
    )
    parser.add_argument(
        '--compresslevel',
        type=int,
        default=1,
        help='Gzip compression level (0-9). Lower is faster.',
    )
    parser.add_argument(
        '--buffer-size',
        type=int,
        default=1024 * 1024,
        help='Copy buffer size in bytes. Larger may speed up IO.',
    )
    
    args = parser.parse_args()
    
    convert_nii_to_niigz(
        args.directory,
        args.output,
        brats_ped_flatten=args.brats_ped_flatten,
        workers=args.workers,
        compresslevel=args.compresslevel,
        buffer_size=args.buffer_size,
    )
