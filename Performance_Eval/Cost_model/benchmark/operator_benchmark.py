"""
Operator Benchmark - 单算子实测实现

Provides real benchmark functionality using PyTorch benchmark tools.
"""

import torch
import torch.fx as fx
import torch.utils.benchmark as benchmark
from typing import Dict, Any, Tuple
from numpy import double


class OperatorBenchmark:
    """
    单算子实测类
    
    使用 PyTorch benchmark 对计算算子进行实际测量。
    """
    # 需要忽略的 kwargs（元数据相关）
    IGNORED_META_KWARGS = {'layer_Rank', 'Stage'}
    
    # 默认配置常量
    DEFAULT_MIN_RUN_TIME_SEC = 0.01

    def __init__(self, verbose: bool = False):
        """
        Args:
            verbose: If True, print detailed benchmark information.
        """
        self.verbose = verbose
        # self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device("cpu")
    def benchmark(
        self,
        node: fx.Node,
        dummy_args: Tuple[Any, ...],
        dummy_kwargs: Dict[str, Any]
    ) -> Tuple[bool, float]:
        """
        对单个节点执行基准测试（核心功能）
        
        Args:
            node: fx.Node to benchmark
            dummy_args: 替换后的参数列表
            dummy_kwargs: 替换后的关键字参数字典
            
        Returns:
            Tuple of (success, time_ms)
        """
        min_run_time_sec = self.DEFAULT_MIN_RUN_TIME_SEC

        try:
            # 执行 benchmark
            timer = benchmark.Timer(
                stmt="op(*args, **kwargs)",
                globals={
                    "op": node.target,
                    "args": dummy_args,
                    "kwargs": dummy_kwargs
                },
                label=f"Operator: {node.name}"
            )

            measurement = timer.blocked_autorange(min_run_time=min_run_time_sec)

            if self.device == "cuda":
                torch.cuda.synchronize()

            # 返回时间单位为秒
            mean_time_sec = measurement.mean  # 不再乘以1000，保持秒为单位

            if self.verbose:
                # 不再转换为毫秒，只输出秒
                stddev_sec = measurement.stddev if hasattr(measurement, "stddev") else 0.0
                print(f"[Benchmark] {node.name}: {mean_time_sec:.4f} sec (stddev: {stddev_sec:.4f} sec)")

            return True, float(mean_time_sec)

        except (ValueError, RuntimeError) as e:
            # 参数准备错误或运行时错误
            if self.verbose:
                print(f"[Warning] Benchmark failed for {node.name}: {e}")
            return False, 0.0
        except Exception as e:
            # 其他未预期的错误
            if self.verbose:
                print(f"[Error] Unexpected error during benchmark for {node.name}: {e}")
            return False, 0.0


