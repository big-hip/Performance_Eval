import zmq
import pickle
import torch
import operator
import time
import sys
import traceback
import gc
from typing import Any, Tuple, Dict, Optional, Callable
from functools import lru_cache

# 尝试导入 NPU 支持
try:
    import torch_npu
except ImportError:
    pass

# ==============================================================================
# 1. 设备上下文管理 (Device Context Abstraction)
# ==============================================================================
class DeviceContext:
    """
    封装设备特定的操作（同步、计时事件、清空缓存），
    避免在主循环中重复进行 if-else 检测。
    """
    def __init__(self):
        self.device_type = 'cpu'
        self.device_str = 'cpu'
        self.event_cls = None
        self._sync_func = lambda: None
        self._empty_cache_func = lambda: None

        if hasattr(torch, 'npu') and torch.npu.is_available():
            self.device_type = 'npu'
            self.device_str = 'npu:0'
            self.event_cls = torch.npu.Event
            self._sync_func = torch.npu.synchronize
            self._empty_cache_func = torch.npu.empty_cache
            # print(f"✅ [DeviceContext] Activated: NPU ({torch.npu.get_device_name(0)})")
        elif torch.cuda.is_available():
            self.device_type = 'cuda'
            self.device_str = 'cuda:0'
            self.event_cls = torch.cuda.Event
            self._sync_func = torch.cuda.synchronize
            self._empty_cache_func = torch.cuda.empty_cache
            # print(f"✅ [DeviceContext] Activated: CUDA ({torch.cuda.get_device_name(0)})")
        else:
            pass
            # print(f"⚠️ [DeviceContext] Activated: CPU only")

    def synchronize(self):
        self._sync_func()

    def empty_cache(self):
        self._empty_cache_func()

    def create_tensor(self, shape, dtype) -> torch.Tensor:
        # 使用 ones 避免除零错误，不计算梯度以节省显存
        return torch.ones(shape, dtype=dtype, device=self.device_str).requires_grad_(False)

    def get_timer_events(self):
        if self.event_cls:
            return self.event_cls(enable_timing=True), self.event_cls(enable_timing=True)
        return None, None

# 全局单例
CTX = DeviceContext()

# ==============================================================================
# 2. 核心 Benchmark 类
# ==============================================================================
class OperatorBenchmark:
    """
    对单一 Torch FX 节点执行算子基准测试。
    """
    WARMUP_ITERS = 5
    RUN_ITERS = 20
    IGNORED_KWARGS = {"layer_Rank", "Stage", "sharding_spec"} # 扩充常见的不需要的参数

    def __init__(self, verbose: bool = True):
        self.verbose = verbose

    @lru_cache(maxsize=1024)
    def _resolve_target(self, target_name: str) -> Optional[Callable]:
        """
        解析算子目标，增加缓存以提升重复算子的处理速度。
        """
        if hasattr(operator, target_name):
            return getattr(operator, target_name)
        if hasattr(torch, target_name):
            return getattr(torch, target_name)
        
        # 复杂路径解析 (e.g. torch.ops.aten.add.Tensor)
        if "." in target_name:
            parts = target_name.split(".")
            obj = torch
            try:
                for part in parts:
                    if part == 'torch': continue
                    obj = getattr(obj, part)
                return obj
            except AttributeError:
                pass
            
            # 尝试 aten ops
            if len(parts) >= 2:
                op_name, op_variant = parts[0], parts[1]
                if hasattr(torch.ops.aten, op_name):
                    aten_op = getattr(torch.ops.aten, op_name)
                    if hasattr(aten_op, op_variant):
                        return getattr(aten_op, op_variant)
        return None

    def _format_arg_summary(self, args, kwargs) -> str:
            if not self.verbose: return ""

            def simple_fmt(x):
                # 1.如果是 Tensor，打印 Shape 和 Dtype
                if isinstance(x, torch.Tensor):
                    shape_str = str(list(x.shape))
                    dtype_str = str(x.dtype).replace('torch.', '')
                    return f"Tensor({shape_str}, {dtype_str})"
                
                # 2. 【关键修改】如果是列表或元组，递归打印内部内容
                elif isinstance(x, (list, tuple)):
                    inner = ", ".join([simple_fmt(item) for item in x])
                    return f"[{inner}]"
                
                # 3.如果是基础类型，直接显示值
                elif isinstance(x, (int, float, str, bool)):
                    return str(x)
                    
                # 4.其他情况打印类型名
                return str(type(x).__name__)
            
            arg_str = ", ".join([simple_fmt(a) for a in args])
            
            # 如果 kwargs 也有内容，顺便打印出来
            if kwargs:
                kwarg_str = ", ".join([f"{k}={simple_fmt(v)}" for k, v in kwargs.items()])
                return f"Args: {arg_str} | Kwargs: {kwarg_str}"
                
            return f"Args: {arg_str}"

    def benchmark(self, node: Any, dummy_args: Tuple[Any, ...], dummy_kwargs: Dict[str, Any]) -> Tuple[bool, float]:
        target_str = str(node.target)
        try:
            # 1. 解析函数
            op_func = self._resolve_target(target_str)
            filtered_kwargs = {k: v for k, v in dummy_kwargs.items() if k not in self.IGNORED_KWARGS}

            func_to_run = None
            run_args = dummy_args
            run_kwargs = filtered_kwargs
            real_op_name = target_str

            # 2. 确定调用方式 (Function vs Method)
            if op_func is not None:
                real_op_name = getattr(op_func, "__name__", target_str)
                func_to_run = op_func
            elif node.op == "call_method":
                method_name = target_str
                obj = dummy_args[0]
                if not hasattr(obj, method_name):
                    raise RuntimeError(f"Object {type(obj)} has no method: {method_name}")
                func_to_run = getattr(obj, method_name)
                run_args = dummy_args[1:] # self 是 obj，从 args 移除
                real_op_name = f"Tensor.{method_name}"
            else:
                raise RuntimeError(f"Unresolvable target: {target_str}")

            if self.verbose:
                print("-" * 60)
                print(f"[Run] Node: {node.name} | Op: {real_op_name}")
                print(f"[Run] {self._format_arg_summary(dummy_args, dummy_kwargs)}")

            # 3. 执行计时
            CTX.synchronize() # 预同步，确保之前的操作完成

            # Warmup
            for _ in range(self.WARMUP_ITERS):
                func_to_run(*run_args, **run_kwargs)
            
            CTX.synchronize() # Warmup 结束同步

            # Timing Run
            start_event, end_event = CTX.get_timer_events()
            
            if start_event:
                # GPU/NPU 计时路径
                start_event.record()
                for _ in range(self.RUN_ITERS):
                    func_to_run(*run_args, **run_kwargs)
                end_event.record()
                CTX.synchronize() # 等待 Event 记录完成
                total_ms = start_event.elapsed_time(end_event)
                mean_time_sec = (total_ms / self.RUN_ITERS) / 1000.0
            else:
                # CPU 计时路径
                start_t = time.perf_counter()
                for _ in range(self.RUN_ITERS):
                    func_to_run(*run_args, **run_kwargs)
                end_t = time.perf_counter()
                mean_time_sec = (end_t - start_t) / self.RUN_ITERS

            if self.verbose:
                print(f"[Result] {mean_time_sec * 1e6:.2f} us")

            return True, mean_time_sec

        except Exception as e:
            if self.verbose:
                pass
                # print(f"❌ [Exec Error] {node.name}: {str(e)}")
                # traceback.print_exc() # 可选：打印详细堆栈
            return False, 0.0
        finally:
            # 这里的 finally 并不一定需要 empty_cache，频繁调用会慢。
            # 放在 Server 循环末尾调用比较好。
            pass

# ==============================================================================
# 3. 数据还原逻辑 (Helper)
# ==============================================================================
class DataReconstructor:
    @staticmethod
    def _str_to_dtype(dtype_str: str):
        # 移除可能的前缀
        clean_str = dtype_str.replace('torch.', '')
        if hasattr(torch, clean_str):
            return getattr(torch, clean_str)
        raise ValueError(f"Unknown dtype string: {dtype_str}")

    @classmethod
    def reconstruct(cls, arg):
        if isinstance(arg, (list, tuple)):
            return type(arg)(cls.reconstruct(x) for x in arg)
        elif isinstance(arg, dict):
            # 检测是否为 Tensor Metadata
            if 'shape' in arg and 'dtype' in arg:
                try:
                    dtype = cls._str_to_dtype(arg['dtype'])
                    return CTX.create_tensor(arg['shape'], dtype)
                except Exception as e:
                    pass
                    # print(f"\n❌ [Data Error] Failed to create tensor: {arg}")
                    # raise e
            return {k: cls.reconstruct(v) for k, v in arg.items()}
        else:
            return arg

# ==============================================================================
# 4. Mock Node (保持不变)
# ==============================================================================
class MockNode:
    def __init__(self, target_str, name_str, op_type='call_function'):
        self.target = target_str 
        self.name = name_str
        self.op = op_type

# ==============================================================================
# 5. Server 通信逻辑
# ==============================================================================
class BenchmarkServer:
    def __init__(self, port=5588):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        # 设置 Linger 为 0，防止 Ctrl+C 时 socket 卡死
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.bind(f"tcp://*:{port}")
        
        self.benchmarker = OperatorBenchmark(verbose=True)
        print(f"🚀 Benchmark Server Running on {CTX.device_type.upper()} | Port: {port}")

    def start(self):
        print(">> Waiting for requests...")
        while True:
            try:
                # 1. 接收
                msg = self.socket.recv()
                payload = pickle.loads(msg)
                
                op_name = payload.get('op', 'unknown')
                node_name = payload.get('name', 'remote_node')
                
                # 2. 构造 Node 和参数
                mock_node = MockNode(op_name, node_name, op_type='call_function')
                
                # 使用专门的重构器，若出错会抛出异常中断本次测试，但被 except 捕获
                real_args = DataReconstructor.reconstruct(payload['args'])
                real_kwargs = DataReconstructor.reconstruct(payload['kwargs'])

                # 3. 执行
                success, cost_time = self.benchmarker.benchmark(
                    mock_node, 
                    tuple(real_args), 
                    real_kwargs
                )

                # 4. 回复
                resp = pickle.dumps({'success': success, 'time': cost_time})
                self.socket.send(resp)

            except Exception as e:
                pass
                # print(f"❌ [Server Loop Error] {e}")
                # traceback.print_exc()
                
                # 关键：确保 Send 被调用，否则 Client 会一直等待 recv 导致死锁
                try:
                    err_resp = pickle.dumps({'success': False, 'time': 0.0, 'error': str(e)})
                    self.socket.send(err_resp)
                except zmq.ZMQError:
                    # 如果 send 也失败（比如 socket 状态错误），通常需要重置 socket
                    print("⚠️ Critical ZMQ Error during error reporting.")
            
            finally:
                # 5. 清理 (防止 OOM)
                # 每次请求后简单清理引用
                del msg, payload
                if 'real_args' in locals(): del real_args
                if 'real_kwargs' in locals(): del real_kwargs
                
                # NPU/CUDA 显存清理：
                # 频繁 empty_cache 会导致性能下降，但 Benchmark 场景下稳定性优先
                # 如果发现太慢，可以加计数器，每 10 次请求清理一次
                # CTX.empty_cache() 
                pass

if __name__ == "__main__":
    # 设置 Python 垃圾回收阈值，稍微激进一点防止 Tensor 泄露
    gc.set_threshold(700, 10, 10)
    
    try:
        server = BenchmarkServer()
        server.start()
    except KeyboardInterrupt:
        print("\n🛑 Server shutting down...")
    except Exception as e:
        print(f"🛑 Fatal Error: {e}")