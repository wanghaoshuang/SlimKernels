import numpy as np

def quantize_to_uint2(matrix: np.ndarray, group_size: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """
    Quantize input matrix to uint2 (0-3 range) by columns in groups
    
    Args:
        matrix: Input matrix as numpy array
        group_size: Number of elements per group, default is 8
        
    Returns:
        tuple(quantized_matrix, scales):
            - quantized_matrix: Quantized matrix of type uint8, range 0-3
            - scales: Quantization scale factors, shape (num_groups, num_cols)
    """
    if not isinstance(matrix, np.ndarray):
        raise TypeError("Input must be a numpy array")
        
    num_rows, num_cols = matrix.shape
    if num_rows % group_size != 0:
        raise ValueError(f"Number of rows ({num_rows}) must be divisible by group_size ({group_size})")
    
    num_groups = num_rows // group_size
    
    # Reshape matrix for group operations
    reshaped = matrix.reshape(num_groups, group_size, num_cols)
    
    # Calculate maximum absolute value per group as quantization scale
    scales = np.max(np.abs(reshaped), axis=1)
    # Avoid division by zero
    scales = np.maximum(scales, 1e-9)
    
    # 对于uint2，量化范围是0-3，有效范围是1.5
    # 因此需要将scales除以1.5以适应这个范围
    num_bits = 2
    quant_range = 2**num_bits - 1  # 3 for uint2
    effective_range = quant_range / 2  # 1.5 for uint2
    scales = scales / effective_range
    
    # Broadcast scales to match reshaped dimensions
    scales_broadcasted = scales[:, np.newaxis, :]
    
    zero_point = 1
    print(reshaped / scales_broadcasted)
    print(np.round(reshaped / scales_broadcasted))
    quantized = np.clip(np.round(reshaped / scales_broadcasted) + zero_point, 0, quant_range).astype(np.uint8)
    
    # Reshape back to original dimensions
    quantized = quantized.reshape(-1, num_cols)
    # Truncate to original number of rows
    quantized = quantized[:num_rows]
    
    return quantized, scales

def dequantize_from_uint2(quantized: np.ndarray, scales: np.ndarray, group_size: int = 8) -> np.ndarray:
    """
    Dequantize uint2 (0-3 range) matrix back to FP16 by columns in groups
    
    Args:
        quantized: Quantized matrix of type uint8, range 0-3
        scales: Quantization scale factors, shape (num_groups, num_cols)
        group_size: Number of elements per group, default is 8
        
    Returns:
        Dequantized matrix as float16 numpy array
    """
    if not isinstance(quantized, np.ndarray) or not isinstance(scales, np.ndarray):
        raise TypeError("Both inputs must be numpy arrays")
    
    if quantized.dtype != np.uint8:
        raise TypeError("Quantized matrix must be of type uint8")
        
    num_rows, num_cols = quantized.shape
    if num_rows % group_size != 0:
        raise ValueError(f"Number of rows ({num_rows}) must be divisible by group_size ({group_size})")
    
    num_groups = num_rows // group_size
    expected_scales_shape = (num_groups, num_cols)
    if scales.shape != expected_scales_shape:
        raise ValueError(f"Scales shape {scales.shape} does not match expected shape {expected_scales_shape}")
    
    # Reshape quantized matrix for group operations
    reshaped_quantized = quantized.reshape(num_groups, group_size, num_cols)
    
    # Broadcast scales to match quantized dimensions
    scales_broadcasted = scales[:, np.newaxis, :]
    
    # Dequantize: subtract zero point and multiply by scale
    zero_point = 1
    dequantized = (reshaped_quantized.astype(np.float32) - zero_point) * scales_broadcasted
    
    # Reshape back to original dimensions and convert to float16
    return dequantized.reshape(num_rows, num_cols).astype(np.float16)

