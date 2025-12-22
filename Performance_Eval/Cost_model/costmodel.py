import torch.fx as fx
import torch
import zmq
import pickle
import os
from typing import Dict, Any, Optional
import numpy as np

__all__ = []
class Really_run:
    """
    Cost Model for operator performance evaluation via Remote Benchmark Server.
    支持多进程安全的 ZMQ 连接。
    """
    def __init__(self, link_manager=None, server_addr="tcp://127.0.0.1:5588"):
        self.link_manager = link_manager
        self.server_addr = server_addr
        
        # --- 关键修改：不要在 init 里创建 socket ---
        # 我们只把它们设为 None，等到真正要用的时候再创建
        self.context = None
        self.socket = None

    def _ensure_connection(self):
        """
        惰性初始化：检查 socket 是否存在，不存在则创建。
        这样可以保证每个子进程都有自己独立的 socket 连接。
        """
        if self.socket is None:
            # print(f">> [CostModel] (PID: {os.getpid()}) 正在连接到 {self.server_addr} ...")
            self.context = zmq.Context()
            self.socket = self.context.socket(zmq.REQ)
            self.socket.connect(self.server_addr)

    def __getstate__(self):
        """
        【关键】告诉 pickle：在序列化我的时候，不要带上 context 和 socket。
        """
        state = self.__dict__.copy()
        # 将不可序列化的 ZMQ 对象设为 None
        state['context'] = None
        state['socket'] = None
        return state

    def __setstate__(self, state):
        """
        【关键】反序列化后，确保这些属性存在（虽然是 None）。
        """
        self.__dict__.update(state)
        self.context = None
        self.socket = None

    def get_computation_time(self, node: Any) -> float:
        # 1. 确保连接已建立 (如果是第一次调用，或者刚传到子进程，这里会触发连接)
        self._ensure_connection()

        # ---------------- 下面是原来的逻辑 ----------------
        
        def custom_map_arg(a: Any, fn: Any) -> Any:
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

        def _extract_metadata(arg: Any) -> Any:
            is_task_node = hasattr(arg, 'op') and hasattr(arg, 'meta')
            if is_task_node:
                meta_val = arg.meta.get('val', arg.meta.get('tensor_meta'))
                if meta_val is None: return None
                if hasattr(meta_val, 'shape') and hasattr(meta_val, 'dtype'):
                    shape = tuple(meta_val.shape) if isinstance(meta_val.shape, (list, tuple)) else tuple(meta_val.shape)
                    dtype_str = str(meta_val.dtype).replace('torch.', '')
                    return {'shape': shape, 'dtype': dtype_str, 'type': 'tensor'}
                return meta_val
            return arg 

        try:
            meta_args = custom_map_arg(node.args, _extract_metadata)
            meta_kwargs = custom_map_arg(node.kwargs, _extract_metadata)
            
            op_target = node.target
            if not isinstance(op_target, str):
                if hasattr(op_target, '__name__'):
                    op_target = op_target.__name__
                else:
                    op_target = str(op_target)

            payload = {
                'op': op_target,
                'name': node.name,
                'args': meta_args,
                'kwargs': meta_kwargs
            }

            self.socket.send(pickle.dumps(payload))
            reply_bytes = self.socket.recv()
            result = pickle.loads(reply_bytes)

            if result['success']:
                return result['time'] * 1e6
            else:
                return 2.0

        except Exception as e:
            # print(f"[Warn] Node {node.name} remote benchmark failed: {e}")
            # 出错后为了保险，下次重连
            self.socket = None 
            return 2.0
    def get_communication_time(self,node):
        # return 2.0
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
                self.hardware='A2,A2'
            def predict(self,op:OpRecord):
                self.perf_model=AnalyticalModel(op,self.hardware)
                perf:ZhanluPerfResult=self.perf_model()
                # print(f'perf.op_time')
                # pprint(asdict(perf),sort_dicts=False)
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
        # print(dummy_tensor)
        op_name =str(node.name)
        # print("------通信算子-------")
        # print(op_name)
        op_name=simplify_op_name(op_name)
        # print(op_name)
        # print(result_shape)
        op=get_dummy_op(op_name,[dummy_tensor])
        # shapes = [[125829120],[4096, 16, 192] ]
        # tensors = [get_dummy_tensor(shape, torch.float16) for shape in shapes]
        # op = get_dummy_op('AllGatherKernel', tensors)
        class DummyInstance:
                pass
        op.instance=DummyInstance()
        # print(node.kwargs.get('comm_group'))
        # print(type(node.kwargs.get('comm_group')))
        op.instance.global_rank_list=list(node.kwargs['comm_group'])#数据类型不是list
        # op.instance.global_rank_list=[8,9]
        time=sop.predict(op)
        # print(time)
        return time




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
    def get_computation_time(
        self, 
        node: fx.Node, 
    ) -> double:
        # print("-----prime_node_target-----")
        # print(node.target)
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
                self.hardware='A2,A2'
            def predict(self,op:OpRecord):
                self.perf_model=AnalyticalModel(op,self.hardware)
                perf:ZhanluPerfResult=self.perf_model()
                # print(f'perf.op_time')
                # pprint(asdict(perf),sort_dicts=False)
                return perf.op_time
        sop=SingleOpCostModel()
        dummylist=[]
        for prenode in node.all_input_nodes:
            prenode_tensor_meta =prenode.meta.get('tensor_meta') if hasattr(prenode,'meta') else None
            #下面这个是找pre节点的pre节点，这个不应该这样的
            if prenode_tensor_meta is None and hasattr(prenode,"args"):
                for arg in prenode.args:
                    if hasattr(arg,'meta') and arg.meta.get('tensor_meta') is not None:
                        prenode_tensor_meta=arg.meta['tensor_meta']
                        break
                if prenode_tensor_meta is None:
                    raise ValueError(f'{prenode} has no meta')
                    # pass
            if prenode_tensor_meta:    
                shape=list(prenode_tensor_meta.shape)
                dtype=prenode_tensor_meta.dtype
                dummy_tensor=get_dummy_tensor(shape,dtype)
                dummylist.append(dummy_tensor)

        op_name=str(node.target)
        op_name=simplify_op_name(op_name)
        # print("-----node.target-------")
        # print(op_name)
        op=get_dummy_op(op_name,dummylist)
        time=sop.predict(op)
        return time 
    
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
                self.hardware='A2,A2'
            def predict(self,op:OpRecord):
                self.perf_model=AnalyticalModel(op,self.hardware)
                perf:ZhanluPerfResult=self.perf_model()
                # print(f'perf.op_time')
                # pprint(asdict(perf),sort_dicts=False)
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
        # print(dummy_tensor)
        op_name =str(node.name)
        # print("------通信算子-------")
        # print(op_name)
        op_name=simplify_op_name(op_name)
        # print(op_name)
        # print(result_shape)
        op=get_dummy_op(op_name,[dummy_tensor])
        # shapes = [[125829120],[4096, 16, 192] ]
        # tensors = [get_dummy_tensor(shape, torch.float16) for shape in shapes]
        # op = get_dummy_op('AllGatherKernel', tensors)
        class DummyInstance:
                pass
        op.instance=DummyInstance()
        # print(node.kwargs.get('comm_group'))
        # print(type(node.kwargs.get('comm_group')))
        op.instance.global_rank_list=list(node.kwargs['comm_group'])#数据类型不是list
        # op.instance.global_rank_list=[8,9]
        time=sop.predict(op)
        # print(time)
        return time