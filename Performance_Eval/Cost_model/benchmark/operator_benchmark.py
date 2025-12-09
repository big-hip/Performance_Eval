import torch
import torch.utils.benchmark as benchmark
import operator
from typing import Any, Tuple, Dict

class OperatorBenchmark:
    """
    对单一 Torch FX 节点执行算子基准测试。
    包含详细的 Debug 输出功能，并支持过滤无效 kwargs。
    """

    DEFAULT_MIN_RUN_TIME_SEC = 0.01
    
    # 定义需要忽略的 kwargs key，这些不会传给算子执行
    IGNORED_KWARGS = {"layer_Rank", "Stage"}

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.device = torch.device("cuda")

    # ---------------------------------------------------------
    # 辅助方法：格式化参数信息
    # ---------------------------------------------------------
    def _format_arg_info(self, arg: Any) -> str:
        if isinstance(arg, torch.Tensor):
            shape_info = list(arg.shape)
            dtype_info = str(arg.dtype).replace("torch.", "")
            return f"Tensor(shape={shape_info}, dtype={dtype_info})"
        elif isinstance(arg, (list, tuple)):
            items = [self._format_arg_info(x) for x in arg]
            if isinstance(arg, list):
                return f"[{', '.join(items)}]"
            else:
                return f"({', '.join(items)})"
        return str(arg)

    # ---------------------------------------------------------
    # 核心逻辑：解析算子 Target
    # ---------------------------------------------------------
    def _resolve_target(self, target_name: str):
        if hasattr(operator, target_name):
            return getattr(operator, target_name)
        if hasattr(torch, target_name):
            return getattr(torch, target_name)
        if "." in target_name:
            parts = target_name.split(".")
            if len(parts) >= 2:
                op_name, op_variant = parts[0], parts[1]
                if hasattr(torch.ops.aten, op_name):
                    aten_op = getattr(torch.ops.aten, op_name)
                    if hasattr(aten_op, op_variant):
                        return getattr(aten_op, op_variant)
        return None

    # ---------------------------------------------------------
    # 执行单算子 Benchmark
    # ---------------------------------------------------------
    def benchmark(
        self,
        node: Any,
        dummy_args: Tuple[Any, ...],
        dummy_kwargs: Dict[str, Any],
    ) -> Tuple[bool, float]:
        
        target_str = str(node.target)

        try:
            # 1. 尝试解析 target
            op_func = self._resolve_target(target_str)

            # =========================================================
            # [新增步骤] 过滤不需要的 kwargs (如 layer_Rank, Stage)
            # =========================================================
            # 使用字典推导式过滤掉在 IGNORED_KWARGS 中的 key
            filtered_kwargs = {
                k: v for k, v in dummy_kwargs.items() 
                if k not in self.IGNORED_KWARGS
            }

            # 2. 构造执行环境 globals_dict (注意：这里用的是 filtered_kwargs)
            globals_dict = {
                "args": dummy_args,
                "kwargs": filtered_kwargs
            }

            real_op_name = target_str 

            # 3. 确定 op 对象并调整参数
            if op_func is not None:
                # === 情况 A: call_function / call_module ===
                globals_dict["op"] = op_func
                if hasattr(op_func, "__name__"):
                    real_op_name = op_func.__name__
            else:
                # === 情况 B: call_method (例如 x.add(y)) ===
                if node.op == "call_method":
                    method_name = target_str
                    obj = dummy_args[0]
                    if not hasattr(obj, method_name):
                        raise RuntimeError(f"Tensor no method: {method_name}")
                    
                    real_method = getattr(obj, method_name)
                    globals_dict["op"] = real_method
                    # call_method 的第一个参数是 self，传给 op 时要去掉
                    globals_dict["args"] = dummy_args[1:] 
                    real_op_name = f"Tensor.{method_name}"
                else:
                    raise RuntimeError(f"Unresolvable target: {target_str}")

            # =========================================================
            # [Debug Log] 打印详细信息 (使用过滤后的 filtered_kwargs)
            # =========================================================
            if self.verbose:
                # 1. 提取 args 的 info
                args_info = [self._format_arg_info(a) for a in globals_dict["args"]]
                
                # 2. 提取 kwargs 的 info (只显示传给算子的参数)
                kwargs_info = [f"{k}={self._format_arg_info(v)}" for k, v in filtered_kwargs.items()]
                
                # 3. 拼装完整调用签名
                full_params = ", ".join(args_info + kwargs_info)
                
                print("-" * 60, flush=True)
                print(f"[Debug Exec] Node:      {node.name}", flush=True)
                print(f"[Debug Exec] Stmt:      op(*args, **kwargs)", flush=True)
                print(f"[Debug Exec] Details:   {real_op_name}({full_params})", flush=True)
                print("-" * 60, flush=True)
            # =========================================================

            # 4. 执行 Timer
            stmt = "op(*args, **kwargs)"
            
            timer = benchmark.Timer(
                stmt=stmt,
                globals=globals_dict,
                label=f"Operator: {node.name}"
            )

            measurement = timer.blocked_autorange(
                min_run_time=self.DEFAULT_MIN_RUN_TIME_SEC
            )
            mean_time = float(measurement.mean)

            if self.verbose:
                print(f"[Benchmark] {node.name} = {mean_time*1e6:.2f} us", flush=True)

            return True, mean_time

        except Exception as e:
            if self.verbose:
                print(f"[Error] Benchmark failed ({node.target}): {e}", flush=True)
            return False, 0.0