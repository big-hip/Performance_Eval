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
        print("-----prime_node_target-----")
        print(node.target)
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
            shape=list(prenode_tensor_meta.shape)
            dtype=prenode_tensor_meta.dtype
            dummy_tensor=get_dummy_tensor(shape,dtype)
            dummylist.append(dummy_tensor)
        op_name=str(node.target)
        op_name=simplify_op_name(op_name)
        print("-----node.target-------")
        print(op_name)
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