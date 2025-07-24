import os
import nibabel as nib
from tqdm import tqdm
import argparse
from pathlib import Path

def convert_nii_to_niigz(root_dir, output_dir):
    """
    Converts all .nii files in subdirectories of root_dir to .nii.gz format,
    saving them to a new directory while preserving the folder structure.

    Args:
        root_dir (str): The path to the root directory containing patient folders.
        output_dir (str): The path to the directory where converted files will be saved.
    """
    root_path = Path(root_dir)
    output_path = Path(output_dir)

    for subdir, _, files in os.walk(root_path):
        for file in tqdm(files, desc=f'Processing {os.path.basename(subdir)}'):
            if file.endswith('.nii'):
                nii_path = Path(subdir) / file
                
                # Create the corresponding output path
                relative_path = nii_path.relative_to(root_path)
                nii_gz_path = output_path / relative_path.with_suffix('.nii.gz')

                # Create parent directories if they don't exist
                nii_gz_path.parent.mkdir(parents=True, exist_ok=True)

                if nii_gz_path.exists():
                    print(f'Skipping {nii_path}, .gz already exists in destination.')
                    continue

                try:
                    # Load the .nii file
                    img = nib.load(nii_path)
                    
                    # Save as .nii.gz
                    nib.save(img, nii_gz_path)
                    
                    print(f'Successfully converted {nii_path} to {nii_gz_path}')
                except Exception as e:
                    print(f'Error converting {nii_path}: {e}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert .nii files to .nii.gz format.')
    parser.add_argument('-d', '--directory', type=str, required=True, 
                        help='Path to the root directory containing the subfolders with .nii files.')
    parser.add_argument('-o', '--output', type=str, required=True,
                        help='Path to the output directory to save .nii.gz files.')
    
    args = parser.parse_args()
    
    convert_nii_to_niigz(args.directory, args.output)