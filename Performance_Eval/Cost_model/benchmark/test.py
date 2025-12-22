import torch
import torch_npu
import os

print(f"PID: {os.getpid()}")
# 再次打印确认 python 内部能读到变量
print(f"OPP: {os.environ.get('ASCEND_OPP_PATH')}") 

try:
    # 强制同步一次，有时候能唤醒驱动
    torch.npu.synchronize()
    print("1. NPU Sync OK")
    
    # 分配一个小 Tensor
    x = torch.ones(1, device="npu:0")
    print(f"2. Allocation OK: {x}")
    
    # 做一次计算
    y = x + 1
    print(f"3. Computation OK: {y}")
    
except Exception as e:
    print(f"❌ FATAL ERROR: {e}")