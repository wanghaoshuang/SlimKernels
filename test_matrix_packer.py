import numpy as np
import pytest
from matrix_packer import pack_matrix, unpack_matrix

def test_pack_matrix():
    # 创建测试输入矩阵 (4x8)
    input_matrix = np.array([
        [0, 1, 2, 3, 0, 1, 2, 3],
        [3, 2, 1, 0, 3, 2, 1, 0],
        [1, 1, 1, 1, 2, 2, 2, 2],
        [0, 3, 2, 1, 3, 0, 1, 2]
    ], dtype=np.uint8)
    
    # 预期输出矩阵 (4x2)
    # 第一组4列: [0,1,2,3] -> 0b00011011 = 27
    # 第二组4列: [0,1,2,3] -> 0b00011011 = 27
    expected_output = np.array([
        [27, 27],  # [0,1,2,3] -> 0b00011011
        [228, 228],  # [3,2,1,0] -> 0b11100100
        [85, 170],  # [1,1,1,1] -> 0b01010101, [2,2,2,2] -> 0b10101010
        [57, 198]  # [0,3,2,1] -> 0b00111001, [3,0,1,2] -> 0b11000110
    ], dtype=np.uint8)
    
    result = pack_matrix(input_matrix)
    np.testing.assert_array_equal(result, expected_output)
    
def test_invalid_input():
    # 测试非法数值范围
    invalid_values = np.array([[0, 1, 2, 4]], dtype=np.uint8)  # 4超出了0-3范围
    with pytest.raises(ValueError, match="输入矩阵的值必须在0-3之间"):
        pack_matrix(invalid_values)
    
    # 测试错误的列数
    invalid_shape = np.array([[0, 1, 2]], dtype=np.uint8)  # 列数不是4的倍数
    with pytest.raises(ValueError, match="输入矩阵的列数必须是4的倍数"):
        pack_matrix(invalid_shape)
    
    # 测试错误的数据类型
    invalid_dtype = np.array([[0, 1, 2, 3]], dtype=np.float32)
    with pytest.raises(ValueError, match="输入矩阵必须是uint8类型"):
        pack_matrix(invalid_dtype) 

def test_unpack_matrix():
    # 创建测试输入矩阵 (4x2)
    packed_matrix = np.array([
        [27, 27],    # 0b00011011 -> [0,1,2,3]
        [228, 228],  # 0b11100100 -> [3,2,1,0]
        [85, 170],   # 0b01010101 -> [1,1,1,1], 0b10101010 -> [2,2,2,2]
        [57, 198]    # 0b00111001 -> [0,3,2,1], 0b11000110 -> [3,0,1,2]
    ], dtype=np.uint8)
    
    # 预期输出矩阵 (4x8)
    expected_output = np.array([
        [0, 1, 2, 3, 0, 1, 2, 3],
        [3, 2, 1, 0, 3, 2, 1, 0],
        [1, 1, 1, 1, 2, 2, 2, 2],
        [0, 3, 2, 1, 3, 0, 1, 2]
    ], dtype=np.uint8)
    
    result = unpack_matrix(packed_matrix)
    np.testing.assert_array_equal(result, expected_output)

def test_pack_unpack_roundtrip():
    # 创建随机测试矩阵
    original = np.random.randint(0, 4, (10, 8), dtype=np.uint8)
    
    # 打包然后解包
    packed = pack_matrix(original)
    unpacked = unpack_matrix(packed)
    
    # 验证结果是否与原始矩阵相同
    np.testing.assert_array_equal(original, unpacked)


