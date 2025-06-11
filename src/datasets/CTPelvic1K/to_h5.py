import os
import nibabel as nib
import h5py
import numpy as np
from scipy import ndimage
from tqdm import tqdm

def normalize_ct(ct_data):
    """
    Normalize CT data to [0, 1] range after clipping to [-1000, 2000] HU.
    
    Args:
        ct_data (numpy.ndarray): CT image data in HU
    
    Returns:
        numpy.ndarray: Normalized CT data in [0, 1] range
    """
    # Clip to [-1000, 2000] HU
    ct_clipped = np.clip(ct_data, -1000, 2000)
    
    # Normalize to [0, 1]
    ct_normalized = (ct_clipped + 1000) / 3000
    
    return ct_normalized

def print_image_info(data, name, is_mask=False):
    """
    Print detailed information about an image.
    
    Args:
        data (numpy.ndarray): Image data
        name (str): Name of the image
        is_mask (bool): Whether the image is a mask
    """
    print(f"\n{name} Information:")
    print(f"Shape: {data.shape}")
    print(f"Data type: {data.dtype}")
    if is_mask:
        unique_labels = np.unique(data)
        print(f"Unique labels: {unique_labels}")
        print(f"Label counts: {dict(zip(*np.unique(data, return_counts=True)))}")
    else:
        print(f"Value range: [{data.min():.2f}, {data.max():.2f}]")
        print(f"Mean: {data.mean():.2f}")
        print(f"Std: {data.std():.2f}")

def resample_image(img, target_pixel_dim, is_mask=False):
    """
    Resample image to target pixel dimensions.
    
    Args:
        img (nibabel.Nifti1Image): Input image
        target_pixel_dim (tuple): Target pixel dimensions (x, y, z) in mm
        is_mask (bool): Whether the image is a mask (for printing purposes)
    
    Returns:
        numpy.ndarray: Resampled image data
    """
    # Get current pixel dimensions from affine matrix
    current_pixel_dim = np.sqrt(np.sum(img.affine[:3, :3] ** 2, axis=0))
    
    # Calculate zoom factors
    zoom_factors = current_pixel_dim / target_pixel_dim
    
    # Get image data
    data = img.get_fdata()
    
    # Print original image information
    image_type = "Mask" if is_mask else "CT"
    print(f"\n{image_type} Original Information:")
    print(f"Original shape: {data.shape}")
    print(f"Original pixel dimensions (mm): {current_pixel_dim}")
    print(f"Target pixel dimensions (mm): {target_pixel_dim}")
    print(f"Zoom factors: {zoom_factors}")
    print_image_info(data, "Original", is_mask)
    
    # Resample using scipy's zoom
    order = 0 if is_mask else 3  # 0 for nearest neighbor (mask), 3 for cubic (CT)
    resampled_data = ndimage.zoom(data, zoom_factors, order=order)
    
    # Print resampled image information
    print_image_info(resampled_data, "Resampled", is_mask)
    
    # Normalize CT data if it's not a mask
    if not is_mask:
        resampled_data = normalize_ct(resampled_data)
        print_image_info(resampled_data, "Normalized", is_mask)
    
    return resampled_data

def convert_nii_to_h5(data_dir, mask_dir, output_file, target_pixel_dim=(1.0, 1.0, 1.0)):
    """
    Convert all nii.gz files from data and mask directories to a single h5 file.
    Resample both CT and mask data to target pixel dimensions.
    
    Args:
        data_dir (str): Directory containing CT nii.gz files
        mask_dir (str): Directory containing segmentation mask nii.gz files
        output_file (str): Path to the output h5 file
        target_pixel_dim (tuple): Target pixel dimensions (x, y, z) in mm
    """
    # Get all nii.gz files from data directory
    data_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.nii.gz')])
    mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith('.nii.gz')])
    
    # Verify that we have the same number of files in both directories
    if len(data_files) != len(mask_files):
        raise ValueError(f"Number of files mismatch: {len(data_files)} CT files vs {len(mask_files)} mask files")
    
    print(f"Found {len(data_files)} pairs of files to convert")
    print(f"Target pixel dimensions: {target_pixel_dim} mm")
    
    # Create a single h5 file for all data
    with h5py.File(output_file, 'w') as f:
        # Process each pair of files
        for data_file, mask_file in tqdm(zip(data_files, mask_files), total=len(data_files), desc="Converting files"):
            try:
                # Extract ID from filename
                sample_id = os.path.splitext(os.path.splitext(data_file)[0])[0]
                print(f"\n{'='*50}")
                print(f"Processing sample: {sample_id}")
                print(f"{'='*50}")
                
                # Load CT data
                data_path = os.path.join(data_dir, data_file)
                data_img = nib.load(data_path)
                
                # Load mask data
                mask_path = os.path.join(mask_dir, mask_file)
                mask_img = nib.load(mask_path)
                
                # Get original pixel dimensions
                original_ct_dim = np.sqrt(np.sum(data_img.affine[:3, :3] ** 2, axis=0))
                original_mask_dim = np.sqrt(np.sum(mask_img.affine[:3, :3] ** 2, axis=0))
                
                # Resample both images
                ct_data = resample_image(data_img, target_pixel_dim, is_mask=False)
                mask_data = resample_image(mask_img, target_pixel_dim, is_mask=True)
                
                # Verify data shapes match
                if ct_data.shape != mask_data.shape:
                    print(f"Warning: Shape mismatch for {data_file}")
                    print(f"CT shape: {ct_data.shape}, Mask shape: {mask_data.shape}")
                    continue
                
                # Save resampled data
                f.create_dataset(f'{sample_id}/ct', data=ct_data, compression='gzip')
                f.create_dataset(f'{sample_id}/mask', data=mask_data, compression='gzip')
                
                # Save original and target pixel dimensions as attributes
                f[f'{sample_id}/ct'].attrs['original_pixel_dim'] = original_ct_dim
                f[f'{sample_id}/ct'].attrs['target_pixel_dim'] = target_pixel_dim
                f[f'{sample_id}/mask'].attrs['original_pixel_dim'] = original_mask_dim
                f[f'{sample_id}/mask'].attrs['target_pixel_dim'] = target_pixel_dim
                
            except Exception as e:
                print(f"Error processing {data_file}: {str(e)}")
                continue

if __name__ == "__main__":
    # Example usage
    data_directory = "/ssdArray/hongyou/data/CTPelvic1K/CTPelvic1K_dataset6_data"
    mask_directory = "/ssdArray/hongyou/data/CTPelvic1K/ipcai2021_dataset6_Anonymized"
    output_file = "data/CTPelvic1K/dataset6.h5"
    
    # Set target pixel dimensions (in mm)
    target_pixel_dim = (1.0, 1.0, 1.0)  # 1mm x 1mm x 1mm
    
    convert_nii_to_h5(data_directory, mask_directory, output_file, target_pixel_dim)