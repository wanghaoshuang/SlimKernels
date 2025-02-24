import numpy as np
import pytest
from quantization import quantize_to_uint2, dequantize_from_uint2

def test_basic_quantization():
    """测试基本的量化功能"""
    # 创建一个简单的测试矩阵
    matrix = np.array([
        [1.0, -2.0],
        [3.0, -4.0],
        [5.0, -6.0],
        [7.0, -8.0],
        [9.0, -10.0],
        [11.0, -12.0],
        [13.0, -14.0],
        [15.0, -16.0],
    ])
    
    quantized, scales = quantize_to_uint2(matrix, group_size=8)
    
    # 检查量化后的值是否在正确范围内
    assert quantized.dtype == np.uint8
    assert np.all(quantized >= 0)
    assert np.all(quantized <= 3)
    
    # 检查scales的形状是否正确
    assert scales.shape == (1, 2)  # 1个组，2列

def test_invalid_input():
    """测试无效输入"""
    # 测试非numpy数组输入
    with pytest.raises(TypeError):
        quantize_to_uint2([[1, 2], [3, 4]])
    
    # 测试行数不能被group_size整除的情况
    matrix = np.random.randn(10, 4)  # 10行不能被8整除
    with pytest.raises(ValueError):
        quantize_to_uint2(matrix, group_size=8)

def test_zero_handling():
    """测试零值处理"""
    matrix = np.zeros((8, 2))
    quantized, scales = quantize_to_uint2(matrix)
    
    # 零值应该被量化到中间值
    assert np.all(quantized == 1)
    # scales应该是非零的
    assert np.all(scales > 0)

def test_multiple_groups():
    """测试多个组的情况"""
    matrix = np.random.randn(16, 4)  # 16行，可以分成2组
    quantized, scales = quantize_to_uint2(matrix, group_size=8)
    
    assert scales.shape == (2, 4)  # 2个组，4列
    assert quantized.shape == (16, 4)

def test_extreme_values():
    """测试极端值"""
    matrix = np.array([
        [1e6, -1e6],
        [1e-6, -1e-6],
        [0.0, 0.0],
        [1.0, -1.0],
        [10.0, -10.0],
        [100.0, -100.0],
        [1000.0, -1000.0],
        [10000.0, -10000.0],
    ])
    
    quantized, scales = quantize_to_uint2(matrix)
    
    # 检查量化结果是否在有效范围内
    assert np.all(quantized >= 0)
    assert np.all(quantized <= 3)


def test_dequantize_invalid_input():
    """测试反量化的无效输入处理"""
    quantized = np.array([[0, 1], [2, 3]], dtype=np.uint8)
    scales = np.array([[1.0, 1.0]])
    
    # 测试非numpy数组输入
    with pytest.raises(TypeError):
        dequantize_from_uint2([[0, 1], [2, 3]], scales)
    
    with pytest.raises(TypeError):
        dequantize_from_uint2(quantized, [[1.0, 1.0]])
    
    # 测试错误的quantized类型
    with pytest.raises(TypeError):
        dequantize_from_uint2(quantized.astype(np.float32), scales)
    
    # 测试不匹配的scales形状
    wrong_scales = np.array([[1.0], [1.0]])
    with pytest.raises(ValueError):
        dequantize_from_uint2(quantized, wrong_scales)

def test_dequantize_multiple_groups():
    """测试多组反量化"""
    matrix = np.random.randn(16, 4).astype(np.float32)  # 16行，可以分成2组
    quantized, scales = quantize_to_uint2(matrix, group_size=8)
    dequantized = dequantize_from_uint2(quantized, scales, group_size=8)
    
    assert dequantized.shape == matrix.shape
    assert dequantized.dtype == np.float16 

def test_quantize_to_uint2_specific_values():
    # 创建一个简单的测试矩阵 (8x2)
    test_matrix = np.array([
        [2.0,  -4.0],
        [1.0,  -2.0],
        [0.5,  -1.0],
        [-2.0,  4.0],
        [-1.0,  2.0],
        [-0.5,  1.0],
        [0.0,   0.0],
        [1.5,  -3.0]
    ], dtype=np.float32)

    # 执行量化
    quantized, scales = quantize_to_uint2(test_matrix, group_size=8)

    # 第一列最大绝对值是2.0，第二列是4.0
    # 现在scales被除以了1.5 (effective_range)
    expected_scales = np.array([[2.0*2/3, 4.0*2/3]])
    np.testing.assert_allclose(scales, expected_scales, rtol=1e-5)

    
    expected_quantized = np.array([
        [3, 0], 
        [2, 0], 
        [1, 1], 
        [0, 3],
        [0, 2],
        [1, 1],
        [1, 1],
        [2, 0],
    ], dtype=np.uint8)

    np.testing.assert_array_equal(quantized, expected_quantized) 