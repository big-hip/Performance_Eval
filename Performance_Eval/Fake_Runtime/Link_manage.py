import queue
import torch
import torch.fx as fx
import threading
import networkx as nx
import time
from dataclasses import dataclass
from .Memory_manage import LifeTime, LifeTime_mp
from numpy import double
from ..Cost_model.costmodel import CostModel


@dataclass
class LinkState:
    src: str
    dst: str
    bandwidth: double
    link_type: str
    occupied: bool = False
    latency: double = 0.0


class Link:
    """
        Manage network topology and communication link states.
        Each Link object corresponds to one device's communication manager thread.
    """
    def __init__(self, topo_graph: nx.Graph):
        self.topo_graph = topo_graph
        self.link_states = self._init_link_states(topo_graph)


    # --------------------------------------------------------
    # --- Initialization & link management ---
    # --------------------------------------------------------
    def _init_link_states(self, G: nx.Graph):
        link_states = []
        for u, v, attr in G.edges(data=True):
            link_states.append(LinkState(
                src=u,
                dst=v,
                bandwidth=attr.get("bandwidth", 0.0),
                link_type=attr.get("link_type", "unknown"),
                latency=self._default_latency(attr.get("link_type", "unknown"))
            ))
        return link_states

    def _default_latency(self, link_type: str):
        mapping = {
            "HCCS": 0.05,
            "PCIe": 0.3,
            "ethernet": 1.0,
            "NVLink": 0.1
        }
        return mapping.get(link_type, 1.0)

    def _get_link_state(self, u: str, v: str):
        """根据节点对查找对应的链路状态"""
        for link in self.link_states:
            if {link.src, link.dst} == {u, v}:
                return link
        return None

    # --------------------------------------------------------
    # --- Link status query & update ---
    # --------------------------------------------------------
    def check_link_ready(self, src: str, dst: str) -> bool:
        """
            Check if there exists a path from src to dst,
            and all links along the path are idle.
        """
        try:
            path = nx.shortest_path(self.topo_graph, source=src, target=dst)
        except nx.NetworkXNoPath:
            return False

        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            link = self._get_link_state(u, v)
            if link is None or link.occupied:
                return False
        return True

    def occupy_link(self, src: str, dst: str):
        """Occupy all links along src→dst path."""
        path = nx.shortest_path(self.topo_graph, source=src, target=dst)
 
        for i in range(len(path) - 1):
            link = self._get_link_state(path[i], path[i + 1])
            if link:
                link.occupied = True

    def release_link(self, src: str, dst: str):
        """Release all links along src→dst path."""
        path = nx.shortest_path(self.topo_graph, source=src, target=dst)
 
        for i in range(len(path) - 1):
            link = self._get_link_state(path[i], path[i + 1])
            if link:
                link.occupied = False

    # --------------------------------------------------------
    # --- Communication readiness check ---
    # --------------------------------------------------------
    def check_comm_data_ready(self, node: fx.Node, mem: dict) -> bool:
        """
            Check whether the data required for communication is ready.
            This ensures LifeTime(wait=False) for all inputs.
        """
        if isinstance(node, fx.Node) :
            if node.all_input_nodes :
                for inp in node.all_input_nodes:
                    if inp.name not in mem:
                        return False
                    elif isinstance(mem[inp.name], LifeTime) and mem[inp.name].wait:
                        return False
        return True
    
    def check_comm_data_ready_mp(self, node: fx.Node, mem: dict) -> bool:
        """
            Check whether the data required for communication is ready.
            This ensures LifeTime(wait=False) for all inputs.
        """
        if isinstance(node, fx.Node) :
            if node.all_input_nodes :
                for inp in node.all_input_nodes:
                    if inp.name not in mem:
                        return False
                    elif isinstance(mem[inp.name], LifeTime_mp) and mem[inp.name].wait == True:
                        return False
                    elif isinstance(mem[inp.name], LifeTime_mp) and mem[inp.name].ready == False:
                        return False
        return True

    # --------------------------------------------------------
    # --- Communication cost model ---
    # --------------------------------------------------------
    def get_communication_time(self, node) -> double:
        """
            Estimate communication latency using CostModel.
        """
        # 获取通信类型 (send/recv/allreduce)
        # comm_type = str(node.target)
        # 使用统一代价模型接口
        # cost_model = CostModel(type=comm_type, input=list(node.all_input_nodes))
        comm_time = 1.00
        return comm_time

    # --------------------------------------------------------
    # --- Simulated communication operations ---
    # --------------------------------------------------------
    def simulate_send(self, src: str, dst: str, node: fx.Node, overlap: bool = False) -> double:
        """
            Simulate a point-to-point Send communication between two devices.
            Return simulated time cost.
        """
        # Wait until the path is free
        while not self.check_link_ready(src, dst):
            time.sleep(0.001)
        self.occupy_link(src, dst)

        # Compute communication time
        comm_time = self.get_communication_time(node)
        # 模拟通信重叠：如果 overlap=False 则阻塞
        if not overlap:
            time.sleep(comm_time * 0.001)  # optional: small real delay

        # Release links
        self.release_link(src, dst)
        return comm_time

    def simulate_collective(self, comm_group: list[int], node: fx.Node) -> double:
        """
            Simulate a collective operation (e.g., AllReduce, Broadcast).
            Simplified model: latency = max(link_latencies) + payload/bandwidth_avg
        """
        # 获取所有 pairwise 链路平均带宽和最大延迟
        bandwidths, latencies = [], []
        for i in range(len(comm_group)):
            for j in range(i + 1, len(comm_group)):
                try:
                    path = nx.shortest_path(self.topo_graph, source=comm_group[i], target=comm_group[j])
                    for u, v in zip(path[:-1], path[1:]):
                        link = self._get_link_state(u, v)
                        if link:
                            bandwidths.append(link.bandwidth)
                            latencies.append(link.latency)
                except nx.NetworkXNoPath:
                    continue

        avg_bw = sum(bandwidths) / len(bandwidths) if bandwidths else 1.0
        max_lat = max(latencies) if latencies else 1.0

        # 使用代价模型预测
        cost_model = CostModel(type=node.target, input=list(node.all_input_nodes))
        base_time = cost_model.forward() if hasattr(cost_model, "forward") else float(cost_model)

        # 综合延迟公式
        collective_time = base_time / avg_bw + max_lat
        return collective_time
