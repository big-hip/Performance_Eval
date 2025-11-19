import queue
import torch 
import torch.fx as fx
import threading
from numpy import double 
from ..Cost_model.costmodel import CostModel 
from .Memory_manage import LifeTime, LifeTime_mp
class Compute:
    """
        If the computing unit is idle, it indicates that computation can begin.

    """
    def __init__(self, comp_queue) -> None:
        self.comp_queue = comp_queue
        
    def check_compute_ready(self,node:fx.Node, mem:dict) -> bool:
        """
            Check whether the data required for computation is ready.
            
        """
        if node.all_input_nodes:
            for inp in node.all_input_nodes:
                if inp.name not in mem:
                    return False
                elif isinstance(mem[inp.name], LifeTime) and mem[inp.name].wait:
                    return False
        
        return True

    def check_compute_ready_mp(self,node:fx.Node, mem:dict) -> bool:
        """
            Check whether the data required for computation is ready.
            
        """
        if node.all_input_nodes:
            for inp in node.all_input_nodes:
                if inp.name not in mem:
                    return False
                elif isinstance(mem[inp.name], LifeTime_mp) and mem[inp.name].wait == True:
                    return False
                elif isinstance(mem[inp.name], LifeTime_mp) and mem[inp.name].ready == False:
                    return False
        
        return True
    
    def get_computation_time(self,node:fx.Node) -> double:
        """
            Get the computation time of this node.
            
        """
        #cost_model = CostModel(type=node.target, input=list(node.all_input_nodes))
        cost_model = 1.00
        return cost_model