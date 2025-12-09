"""
Cost Model - 主接口

整合单算子实测、单算子预估和通信算子的统一接口。
"""

import torch.fx as fx
import torch
from typing import Dict, Any, Optional
from numpy import double


from dataclasses import asdict
from pprint import pprint

from custom import op_costmodel
from zhanlu.backend.analytical_model import AnalyticalModel
from zhanlu.backend.perf_result import ZhanluPerfResult
from zhanlu.frontend.utils.op_record import OpRecord
from zhanlu.frontend.utils.tensor_record import TensorRecord
from zhanlu.frontend.utils.op_record import simplify_op_name
from zhanlu.frontend.utils.op_record import to_camel
from .benchmark import OperatorBenchmark
import numpy as np
import math
class CostModel:
    """
    Cost Model for operator performance evaluation.
    """
    def __init__(self, link_manager=None):
        """
        Args:
            link_manager: Optional Link instance for communication model to access topology.
        """
    # def get_computation_time(
    #     self, 
    #     node: fx.Node, 
    # ) -> double:
    #     print("-----prime_node_target-----")
    #     print(node.target)
    #     def get_dummy_tensor(shape,dtype):
    #         return TensorRecord(
    #             name='tensor',global_shape=shape,local_shape=shape,
    #             type='forward',dtype=dtype,is_dtensor=False,requires_grad=False,
    #             module_path='',module_id=0,device_mesh='',
    #             placements='',producer=[],consumer=[]
    #         )
    #     def get_dummy_op(name,inputs):
    #             return OpRecord(
    #                 id=0,name=name,type='forward',subtype='',comm_type='',
    #                 inputs=inputs,outputs=[],
    #                 module_instance=None,module_path='',module_id=0,
    #                 fusion_type='',raw_name='',instance=None,op_type=''
    #             )
    #     class SingleOpCostModel:
    #         def __init__(self):
    #             self.hardware='A3,A3'
    #         def predict(self,op:OpRecord):
    #             self.perf_model=AnalyticalModel(op,self.hardware)
    #             perf:ZhanluPerfResult=self.perf_model()
    #             print(f'perf.op_time')
    #             pprint(asdict(perf),sort_dicts=False)
    #             return perf.op_time
    #     sop=SingleOpCostModel()
    #     dummylist=[]
    #     for prenode in node.all_input_nodes:
    #         prenode_tensor_meta =prenode.meta.get('tensor_meta') if hasattr(prenode,'meta') else None
    #         #下面这个是找pre节点的pre节点，这个不应该这样的
    #         if prenode_tensor_meta is None and hasattr(prenode,"args"):
    #             for arg in prenode.args:
    #                 if hasattr(arg,'meta') and arg.meta.get('tensor_meta') is not None:
    #                     prenode_tensor_meta=arg.meta['tensor_meta']
    #                     break
    #             if prenode_tensor_meta is None:
    #                 raise ValueError(f'{prenode} has no meta')
    #         shape=list(prenode_tensor_meta.shape)
    #         dtype=prenode_tensor_meta.dtype
    #         dummy_tensor=get_dummy_tensor(shape,dtype)
    #         dummylist.append(dummy_tensor)
    #     op_name=str(node.target)
    #     op_name=simplify_op_name(op_name)
    #     print("-----node.target-------")
    #     print(op_name)
    #     op=get_dummy_op(op_name,dummylist)
    #     time=sop.predict(op)
    #     return time 
    
    # def get_computation_time(
    #         self, 
    #         node: fx.Node, 
    #     ) -> float:
    #     # print(f"\n>> [Debug] 进入函数处理节点: {node.name}", flush=True)

    #     # 你的防御性代码 (虽然建议放在 __init__)
    #     if not hasattr(self, 'benchmark'):
    #         self.benchmark = OperatorBenchmark(verbose=False)
    #     if not hasattr(self, 'device'):
    #         self.device = torch.device("cpu")

    #     # --- 定义替换函数 ---
    #     def _substitute(arg: Any) -> Any:
    #         # 1. 调试打印
    #         # flush=True 确保在多进程或重定向时能立即看到输出
    #         print("  [Step] 1. Checking arg...", flush=True)
    #         print(f"  [Info] Node args: {node.args}", flush=True)

    #         # 2. 如果 arg 是 fx.Node (前继节点)
    #         if isinstance(arg, fx.Node):
    #             print("  [Step] 2. Arg is a Node, fetching meta...", flush=True)
                
    #             # 获取前继节点的元数据
    #             # 优先尝试 'val' (TorchInductor/ShapeProp 标准 key)
    #             # 其次尝试 'tensor_meta' (旧版或其他 Pass)
    #             meta_val = arg.meta.get('val', arg.meta.get('tensor_meta'))
                
    #             if meta_val is None:
    #                 print(f"  [Warn] Node {arg.name} has NO meta info! Skipping.", flush=True)
    #                 return None 

    #             # 3. 判断元数据类型并构造数据
    #             if isinstance(meta_val, torch.Tensor):
    #                 print(f"  [Action] Creating Fake Input from meta shape: {meta_val.shape}", flush=True)
    #                 # 构造同形状、同类型、同设备的真实 Tensor
    #                 return torch.zeros(
    #                     size=meta_val.shape, 
    #                     dtype=meta_val.dtype, 
    #                     device="cpu"
    #                 )
    #             else:
    #                 # 如果前继节点输出 int/float/list，直接返回
    #                 print(f"  [Action] Returning non-tensor value: {meta_val}", flush=True)
    #                 return meta_val
            
    #         # 3. 如果 arg 不是节点（字面量），直接返回
    #         return arg 

    #     # --- 执行映射和测速 ---
    #     try:
    #         # 4. 映射参数
    #         # fx.map_arg 会自动遍历 args (包括列表、元组里的 node) 并调用 _substitute
    #         dummy_args = fx.map_arg(node.args, _substitute)
    #         print(dummy_args,flush=True)
    #         dummy_kwargs = fx.map_arg(node.kwargs, _substitute)
    #         print(dummy_kwargs,flush=True)
    #         # 检查是否构造成功 (如果 _substitute 返回 None，map_arg 结果里会有 None)
    #         # 这里简单做个打印
    #         # print(f"  [Ready] Dummy Args Created: {dummy_args}")

    #         # 5. 执行测速 (你代码里注释掉了，这里我放开演示逻辑)
    #         # success, time_ms = self.benchmark.benchmark(node, dummy_args, dummy_kwargs)
            
    #         # if success:
    #         #     return time_ms
    #         # return 0.0
            
    #         print("  [Done] Mock execution finished.", flush=True)
    #         return 2.0 # 按照你的代码返回 2

    #     except Exception as e:
    #         print(f"[Warn] Node {node.name} benchmark failed: {e}", flush=True)
    #         import traceback
    #         traceback.print_exc()
    #         return 0.0
    
    
    def get_computation_time(
            self, 
            node: Any, # 这里传入的是 TaskNode
        ) -> float:
        
        # print(f"\n>> [Debug] CostModel 处理节点: {node.name}", flush=True)

        # 1. 初始化环境
        if not hasattr(self, 'benchmark'):
            self.benchmark = OperatorBenchmark(verbose=True)
        print(f"\n>> [Debug] 进入函数处理节点: {node.name}", flush=True)
        # --- 定义一个简单的递归映射工具 (替代 fx.map_arg) ---
        def custom_map_arg(a: Any, fn: Any) -> Any:
            """
            递归遍历参数结构 (list, tuple, dict)，对叶子节点应用 fn。
            完全替代 torch.fx.map_arg，不依赖 fx。
            """
            if isinstance(a, (tuple, list)):
                return type(a)(custom_map_arg(x, fn) for x in a)
            elif isinstance(a, dict):
                return {k: custom_map_arg(v, fn) for k, v in a.items()}
            elif isinstance(a, slice):
                return slice(custom_map_arg(a.start, fn), 
                             custom_map_arg(a.stop, fn), 
                             custom_map_arg(a.step, fn))
            else:
                return fn(a)

        # --- 定义替换逻辑 (把 TaskNode 变成 Dummy Tensor) ---
        def _substitute(arg: Any) -> Any:
            # 判断 arg 是否为 TaskNode
            # 因为 TaskNode 是自定义类，我们通过检查属性来“鸭子类型”判断
            is_task_node = hasattr(arg, 'op') and hasattr(arg, 'meta') and hasattr(arg, 'target')
            
            if is_task_node:
                # print(f"  [Step] Found TaskNode dependency: {arg.name}", flush=True)
                
                # 获取元数据 (我们在转换阶段已经把 FakeTensorMeta 放进去了)
                # 优先找 'val'，其次 'tensor_meta'
                meta_val = arg.meta.get('val', arg.meta.get('tensor_meta'))
                
                if meta_val is None:
                    # print(f"  [Warn] Node {arg.name} has NO meta info! Skipping.", flush=True)
                    return None 

                # 处理 FakeTensorMeta (有 shape 和 dtype 属性的对象)
                if hasattr(meta_val, 'shape') and hasattr(meta_val, 'dtype'):
                    shape = meta_val.shape
                    dtype = meta_val.dtype
                    
                    # 确保 shape 是 tuple (TaskNode 传输过来可能是 list)
                    if isinstance(shape, list):
                        shape = tuple(shape)
                        
                    # 构造 Dummy Tensor 用于 benchmark
                    # 注意：这里我们真的创建了一个 tensor，以便算子能跑起来
                    return torch.zeros(
                        size=shape, 
                        dtype=dtype, 
                        device="cpu" 
                    )
                
                # 如果 meta_val 本身就是 int/float 等标量（有些 shape prop 会这样存）
                return meta_val
            
            # 如果不是 Node (是 int, float, str 等字面量)，直接返回
            return arg 

        # --- 执行主逻辑 ---
        try:
            # 2. 映射 args 和 kwargs
            # 使用我们手写的 custom_map_arg 替代 fx.map_arg
            dummy_args = custom_map_arg(node.args, _substitute)
            dummy_kwargs = custom_map_arg(node.kwargs, _substitute)
            # print(dummy_args )
            # print(dummy_kwargs)
            # print(f"  [Debug] Prepared Args for {node.target}: {dummy_args}", flush=True)

            # 3. 调用 Benchmark (如果有的话)
            # 注意：node.target 在 TaskNode 里是字符串 (如 "add")
            # OperatorBenchmark 需要能够处理字符串 target，或者你需要在这里做一个简单的字符串到 torch.ops 的映射
            # 如果你的 benchmark 库支持字符串 op name，那就直接传
            print("准备执行函数了")
            success, time = self.benchmark.benchmark(node, dummy_args, dummy_kwargs)
            time_us=time* 1e6   # 转成微秒
            print("----success-----")
            print(success)
            if success:
                print("----time----")
                print(time_us)
                return time_us
            
            # print("  [Done] Mock execution finished.", flush=True)
            return 2.0 # 返回 Mock 值

        except Exception as e:
            print(f"[Warn] Node {node.name} benchmark failed: {e}", flush=True)
            # import traceback
            # traceback.print_exc()
            return 0.0
        
    def get_communication_time(self,node):
        def get_dummy_tensor(shape,dtype):
                return TensorRecord(
                    name='tensor',global_shape=shape,local_shape=shape,
                    type='forward',dtype=dtype,is_dtensor=False,requires_grad=False,
                    module_path='',module_id=0,device_mesh='',
                    placements='',producer=[],consumer=[]
                )
        def get_dummy_op(name,inputs):
                return OpRecord(
                    id=0,name=name,type='forward',subtype='',comm_type='',
                    inputs=inputs,outputs=[],
                    module_instance=None,module_path='',module_id=0,
                    fusion_type='',raw_name='',instance=None,op_type=''
            )
        class SingleOpCostModel:
            def __init__(self):
                self.hardware='A3,A3'
            def predict(self,op:OpRecord):
                self.perf_model=AnalyticalModel(op,self.hardware)
                perf:ZhanluPerfResult=self.perf_model()
                print(f'perf.op_time')
                pprint(asdict(perf),sort_dicts=False)
                return perf.op_time
        sop=SingleOpCostModel()
        # 初始化累加器
        total_input_elements = 0
        for prenode in node.all_input_nodes:
            prenode_tensor_meta =prenode.meta.get('tensor_meta') if hasattr(prenode,'meta') else None
            #将所有shape的元素总量求和，并打包成一个Numpy向量
            if prenode_tensor_meta is not None:
                shape = prenode_tensor_meta.shape
                arg_numel= math.prod(shape)
                total_input_elements += arg_numel
        result_shape = [total_input_elements]#之前的两层，是为了后面多的for循环，这里用不到
        dummy_tensor=get_dummy_tensor(result_shape,torch.float32)
        print(dummy_tensor)
        op_name =str(node.name)
        print("------通信算子-------")
        print(op_name)
        op_name=simplify_op_name(op_name)
        print(op_name)
        print(result_shape)
        op=get_dummy_op(op_name,[dummy_tensor])
        # shapes = [[125829120],[4096, 16, 192] ]
        # tensors = [get_dummy_tensor(shape, torch.float16) for shape in shapes]
        # op = get_dummy_op('AllGatherKernel', tensors)
        class DummyInstance:
                pass
        op.instance=DummyInstance()
        print(node.kwargs.get('comm_group'))
        print(type(node.kwargs.get('comm_group')))
        op.instance.global_rank_list=list(node.kwargs['comm_group'])#数据类型不是list
        # op.instance.global_rank_list=[8,9]
        time=sop.predict(op)
        print(time)
        return time