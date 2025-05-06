import torch
import time
import numpy as np

def test_performance(device='cpu', size=10000, iterations=10):
    """测试指定设备(CPU/GPU)的性能"""
    # 确保设备可用
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA不可用，将使用CPU进行测试")
        device = 'cpu'
    
    print(f"\n在{device.upper()}上测试矩阵乘法性能，矩阵大小: {size}x{size}")
    
    # 创建大型随机矩阵
    a = torch.randn(size, size, device=device)
    b = torch.randn(size, size, device=device)
    
    # 预热(确保GPU已经初始化)
    if device == 'cuda':
        for _ in range(5):
            _ = torch.mm(a, b)
        torch.cuda.synchronize()  # 确保所有操作完成
    
    # 测试矩阵乘法性能
    start_time = time.perf_counter()  # 使用更高精度的计时器
    for _ in range(iterations):
        c = torch.mm(a, b)
    
    if device == 'cuda':
        torch.cuda.synchronize()  # 确保所有GPU操作完成
    
    elapsed_time = (time.perf_counter() - start_time) / iterations
    
    # 验证结果没有被优化掉
    checksum = c.sum().item()
    print(f"{device.upper()}平均执行时间: {elapsed_time:.6f}秒")
    print(f"结果校验和: {checksum:.2f} (用于验证计算确实执行)")
    return elapsed_time

def main():
    # 设置矩阵大小(可以根据你的硬件调整)
    matrix_size = 5000  # 对于内存较小的GPU可能需要减小这个值
    
    # 测试CPU性能
    cpu_time = test_performance('cpu', matrix_size)
    
    # 测试GPU性能(如果可用)
    if torch.cuda.is_available():
        gpu_time = test_performance('cuda', matrix_size)
        if gpu_time > 0:
            print(f"\n性能对比: GPU比CPU快 {cpu_time/gpu_time:.2f} 倍")
        else:
            print("\nGPU执行时间过短，无法准确测量加速比")
    else:
        print("\n未检测到可用的CUDA GPU")

if __name__ == "__main__":
    main()