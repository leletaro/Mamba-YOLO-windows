
import torch
import sys

def check_pytorch_cuda():
    """检查PyTorch和CUDA版本信息"""
    print("=" * 50)
    print("PyTorch和CUDA版本检测")
    print("=" * 50)
    
    # 检查PyTorch版本
    print(f"PyTorch版本: {torch.__version__}")
    
    # 检查CUDA可用性
    cuda_available = torch.cuda.is_available()
    print(f"CUDA是否可用: {cuda_available}")
    
    if cuda_available:
        # 获取CUDA版本
        print(f"CUDA版本: {torch.version.cuda}")
        
        # 获取当前CUDA设备信息
        device_count = torch.cuda.device_count()
        print(f"可用GPU数量: {device_count}")
        
        for i in range(device_count):
            device_name = torch.cuda.get_device_name(i)
            print(f"GPU {i}: {device_name}")
            
        current_device = torch.cuda.current_device()
        print(f"当前设备索引: {current_device}")
        
        # 显示设备内存信息
        if device_count > 0:
            memory_allocated = torch.cuda.memory_allocated(current_device) / 1024**2
            memory_reserved = torch.cuda.memory_reserved(current_device) / 1024**2
            print(f"已分配内存: {memory_allocated:.2f} MB")
            print(f"预留内存: {memory_reserved:.2f} MB")
    else:
        print("警告: CUDA不可用，当前使用CPU模式")
    
    # 检查环境变量
    print("\n环境变量信息:")
    import os
    cuda_home = os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH')
    print(f"CUDA_HOME: {cuda_home}")
    
    # Python版本信息
    print(f"Python版本: {sys.version}")
    
    print("\n" + "=" * 50)

if __name__ == "__main__":
    check_pytorch_cuda()
