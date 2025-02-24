import numpy as np

def pack_matrix(input_matrix):
    """
    将输入的uint8矩阵（实际值为uint2）每4列打包为1列uint8
    
    Args:
        input_matrix: numpy array，数据类型为uint8，实际值范围为0-3
        
    Returns:
        numpy array: 打包后的矩阵，每4列原始数据打包为1列
    """
    if input_matrix.dtype != np.uint8:
        raise ValueError("输入矩阵必须是uint8类型")
        
    if input_matrix.shape[1] % 4 != 0:
        raise ValueError("输入矩阵的列数必须是4的倍数")
        
    # 验证输入值范围是否在0-3之间
    if np.any((input_matrix < 0) | (input_matrix > 3)):
        raise ValueError("输入矩阵的值必须在0-3之间")
    
    rows = input_matrix.shape[0]
    cols = input_matrix.shape[1] // 4
    
    # 创建结果矩阵
    result = np.zeros((rows, cols), dtype=np.uint8)
    
    for i in range(cols):
        # 获取每4列
        col_group = input_matrix[:, i*4:(i+1)*4]
        # 使用位运算打包4个2位数到一个8位数
        result[:, i] = (col_group[:, 0] << 6) | (col_group[:, 1] << 4) | \
                      (col_group[:, 2] << 2) | col_group[:, 3]
    
    return result 

def unpack_matrix(packed_matrix):
    """
    将pack_matrix打包的矩阵还原为原始矩阵
    
    Args:
        packed_matrix: numpy array，数据类型为uint8，是pack_matrix的输出结果
        
    Returns:
        numpy array: 还原后的矩阵，每1列解包为4列，值范围为0-3
    """
    if packed_matrix.dtype != np.uint8:
        raise ValueError("输入矩阵必须是uint8类型")
    
    rows = packed_matrix.shape[0]
    cols = packed_matrix.shape[1] * 4
    
    # 创建结果矩阵
    result = np.zeros((rows, cols), dtype=np.uint8)
    
    for i in range(packed_matrix.shape[1]):
        # 获取打包的列
        packed_col = packed_matrix[:, i]
        # 解包每个8位数到4个2位数
        result[:, i*4] = (packed_col >> 6) & 0b11
        result[:, i*4 + 1] = (packed_col >> 4) & 0b11
        result[:, i*4 + 2] = (packed_col >> 2) & 0b11
        result[:, i*4 + 3] = packed_col & 0b11
    
    return result

