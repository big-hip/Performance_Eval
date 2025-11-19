"""
Cost Model - 主接口

整合单算子实测、单算子预估和通信算子的统一接口。
"""

import torch.fx as fx
import torch
from typing import Dict, Any, Optional
from numpy import double

from .benchmark import OperatorBenchmark
from .simulation import OperatorSimulation
from .communication import CommunicationModel


class CostModel:
    """
    Cost Model for operator performance evaluation.
    
    整合了三个模块：
    - Real benchmark for computation operators (单算子实测)
    - Simulation for computation operators (单算子预估)
    - Communication operators (通信算子)
    """
    def __init__(self, link_manager=None):
        """
        Args:
            link_manager: Optional Link instance for communication model to access topology.
        """
        # 初始化三个子模块
        self.benchmark = OperatorBenchmark(verbose=False)
        self.simulation = OperatorSimulation(verbose=False)
        self.communication = CommunicationModel(verbose=False)
        
        # 内部维护 context：存储节点到其输出 tensor 的映射
        self.context: Dict[fx.Node, Any] = {}
        self.device = torch.device("cpu")
    def _create_dummy_tensor(
        self,
        shape: torch.Size, 
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        根据 shape 和 dtype 创建一个用于 benchmark 的随机替身张量。
        """
        if not shape:
            return torch.tensor(0, dtype=dtype, device=self.device)
        if dtype.is_floating_point:
            return torch.randn(shape, dtype=dtype, device=self.device)
        elif dtype == torch.bool:
            return torch.randint(0, 2, shape, dtype=dtype, device=self.device)
        else:
            return torch.randint(0, 100, shape, dtype=dtype, device=self.device)
        
    def _get_tensor_for_node(self,node):
        node_tensor_meta=node.meta.get("tensor_meta") if hasattr(node, "meta") else None
        if node_tensor_meta is None:
            return None
        else:
            shape =list(node_tensor_meta.shape)#或许这个转list是多余的
            dtype =node_tensor_meta.dtype
            device=self.device
            return self._create_dummy_tensor(shape,dtype)
    def update_context(self, node: fx.Node) -> None:
        """
        更新 context，存储节点的输出 tensor

        Args:
            node: fx.Node 节点
        """
        self.context[node] = self._get_tensor_for_node(node)
        # def print_context (context):
        #     """
        #     打印当前 context 中存储的所有节点及其对应的张量信息。
        #     """
        #     if not self.context:
        #         print("Context is empty.")
        #         return

        #     print("Current context contains the following nodes and tensors:")
        #     for node, tensor in context.items():
        #         node_info = f"Node: {node.name}, Op: {node.op}"
        #         if tensor is not None:
        #             print(f"{node_info} -> Tensor Shape: {tensor.shape}, Dtype: {tensor.dtype}")
        #         else:
        #             print(f"{node_info} -> Tensor: None")
        # print_context(self.context)

    
    def clear_context(self) -> None:
        """清空 context"""
        self.context.clear()

    def is_communication_operator(self, node: fx.Node) -> bool:
        """
        判断是否为通信算子
        
        Args:
            node: fx.Node to check
            
        Returns:
            bool: True if it's a communication operator
        """
        return self.communication.is_communication_operator(node)

    def is_placeholder_node(self,node)-> bool:
        """
        判断是否为placeholder节点
        
        Args:
            node: fx.Node to check
            
        Returns:
            bool: True if it's a placeholder node
        """
        print(node.op)
        if node.op == "placeholder":
            return True
        return False

        

    def get_computation_time(
        self, 
        node: fx.Node, 
    ) -> double:
        """
        获取计算算子的执行时间
        
        策略：
        1. 判断是否是通信节点，如果是返回0（由通信算子接口处理）
        2. 从内部 context 中获取前继节点的 tensor，构建 context 字典
        3. 先尝试单算子模拟 self.simulation.simulate(node, context)
        4. 如果模拟失败，再尝试单算子实测 self.benchmark.benchmark(node, context, min_run_time_sec)
        
        Args:
            node: fx.Node 要评估的节点
            predecessor_nodes: Optional[list] 前继节点列表
                    - 如果为 None，使用 node.all_input_nodes 作为前继节点
                    - 从内部 self.context 中查找这些前继节点的输出 tensor
            min_run_time_sec: Optional[float] benchmark 的最小运行时间（秒）
            
        Returns:
            double: Computation time in milliseconds
        """
        #如果是placeholder 先加入context
        if self.is_placeholder_node(node):
            # print("打印所有前繼承節點_placeholder")
            print(node.all_input_nodes)
            self.update_context(node)
        else:
            # 确定前继节点
            # print("打印所有前繼承節點")
            print(node.all_input_nodes)
            if node.all_input_nodes:
                predecessor_nodes = list(node.all_input_nodes)
                # 保证前继节点与tensor映射全部存在 context 中
                for pred_node in predecessor_nodes:
                    if pred_node not in self.context:
                        self.update_context(pred_node)
            # 判断是否是通信节点
            if self.is_communication_operator(node):
                # TODO:调用通讯类的处理方式自己处理
                return 1.0

            # 先尝试单算子模拟
            try:
                pass
                # simulated_time = self.simulation.simulate(node, context if use_context else None)
                # if simulated_time > 0:
                #     return simulated_time
            except Exception:
                # 模拟失败，继续尝试实测
                pass
            
            # 如果模拟失败，尝试单算子实测
            def _substitute(arg: Any) -> Any:
                if isinstance(arg, fx.Node):
                    if arg not in self.context:
                        raise RuntimeError(f"上游节点 {arg.name} 不在 context 中")
                    return self.context[arg]
                return arg 

            dummy_args = fx.map_arg(node.args, _substitute)
            dummy_kwargs = fx.map_arg(node.kwargs, _substitute)
            success, time_ms = self.benchmark.benchmark(node, dummy_args,dummy_kwargs)
            if success:
                return time_ms
            
            # # 实测也失败，返回默认模拟值（不传 context）
            # return self.simulation.simulate(node)

    def simulate_computation_time(self, node: fx.Node) -> double:
        """
        模拟计算算子的执行时间（单算子预估）
        
        Args:
            node: fx.Node to simulate
            
        Returns:
            double: Simulated computation time in milliseconds
        """
        return self.simulation.simulate(node)

    def get_communication_time(
        self,
        node: fx.Node,
        **kwargs
    ) -> double:
        """
        获取通信算子的执行时间
        
        Args:
            node: fx.Node representing communication operator
            **kwargs: Additional parameters for communication modeling
                     - comm_group: List of device IDs
                     - data_size: Size of data (bytes)
                     - comm_type: Type of communication
                     - overlap: Whether can overlap
        
        Returns:
            double: Communication time in milliseconds
        """
        return self.communication.estimate_time(node, **kwargs)


# 注意：不再提供全局默认实例
# 请显式创建 CostModel 实例并传递给需要它的类（Compute, Link, Performance_Evaluation）
