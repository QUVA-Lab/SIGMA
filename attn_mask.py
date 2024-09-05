import numpy as np

def create_binary_kk_attention_mask_2d(grid_size=14, k=5):
    assert k % 2 == 1, "k should be odd to have a symmetric neighborhood"
    num_patches = grid_size**2
    mask = np.zeros((num_patches, num_patches))  # Initialize mask with zeros
    
    # Determine the range offset from the center for the kxk neighborhood
    offset = k // 2
    
    for center_idx in range(num_patches):
        center_row, center_col = divmod(center_idx, grid_size)
        
        # Calculate the start and end indices for the neighborhood in both dimensions
        row_start = max(0, center_row - offset)
        row_end = min(grid_size, center_row + offset + 1)
        col_start = max(0, center_col - offset)
        col_end = min(grid_size, center_col + offset + 1)
        
        # Enable attention within the kxk neighborhood
        for row in range(row_start, row_end):
            for col in range(col_start, col_end):
                neighbor_idx = row * grid_size + col
                mask[center_idx, neighbor_idx] = 1  # Enable attention

    return mask

# Create the binary attention mask for a 3x3 neighborhood within the 14x14 grid
binary_kk_attention_mask_2d = create_binary_kk_attention_mask_2d()

# Visualize a portion of the mask for the first few patches to confirm
# print(binary_kk_attention_mask_2d[:10, :10])