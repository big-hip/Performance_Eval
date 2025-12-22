
from .Simulation import Performance_Evaluation
from numpy import double
import torch
import torch.fx as fx
from torch.fx.experimental.symbolic_shapes import ShapeEnv
from torch.fx import GraphModule
import copy
from Dist_IR import HcclSend
from .Link_manage import Link
from ast import Tuple
class PD_Performance_Eval:
    """
        Class for evaluating the performance of distributed parallelism with Prefill-Decode (PD) separation strategy.
        args:
            original_prefill_FW : The original forward graph of prefill stage.
            prefill_graph : The prefill stage graph after PD separation and parallelism.
            decode_graph : The decode stage graph after PD separation and parallelism.
            inp_seq_len : Input sequence length for the prefill stage.
            res_seq_len : Result sequence length for the decode stage.
        return:
            total_time : The total estimated time for the PD separation strategy.
    """
    def __init__(self, original_prefill_FW, original_decode_FW, prefill_graph, decode_graph, inp_seq_len, res_seq_len, really_run:bool = False):
        self.original_prefill_FW = copy.deepcopy(original_prefill_FW)
        self.original_decode_FW = copy.deepcopy(original_decode_FW)
        self.prefill_graph = prefill_graph
        self.decode_graph = decode_graph
        self.inp_seq_len = inp_seq_len
        self.res_seq_len = res_seq_len
        self.really_run = really_run


    def _get_shape_env(self, gm: GraphModule) -> ShapeEnv:
        for node in gm.graph.nodes:
            if 'tensor_meta' in node.meta:
                for dim in node.meta['tensor_meta'].shape:
                    if isinstance(dim, torch.SymInt) and hasattr(dim, 'node'):
                        return dim.node.shape_env
        for node in gm.graph.nodes:
            if 'val' in node.meta:
                if not isinstance(node.meta['val'], Tuple) and hasattr(node.meta['val'], 'shape'):
                    for dim in node.meta['val'].shape:
                        if isinstance(dim, torch.SymInt) and hasattr(dim, 'node'):
                            return dim.node.shape_env
        return None
    
    def _modify_seq_sym(self, shape_env: ShapeEnv, new_seq_len: int):
        for key, value in shape_env.var_to_val.items():
            if value == self.inp_seq_len:
                shape_env.var_to_val[key] = new_seq_len
                return key
        return None
    

    def Evaluate(self) -> double:
        # Step 1 : Evaluate Prefill Stage
        time1 = 0.0
        P_Stage_Eval = Performance_Evaluation(self.prefill_graph, really_run = self.really_run)
        time1 += P_Stage_Eval.Evaluate()
        print("Prefill Stage Evaluation Time: ", time1)

        # Step 2 : Send-Recv KV Cache, identify node.meta['is_write_kv'] = True
        time2 = 0.0
        kv_nodes = []
        for node in self.original_prefill_FW.graph.nodes:
            if 'is_write_kv' in node.meta:
                if node.meta['is_write_kv'] == True:
                    kv_nodes.append(node)
                    self._update_graph_shapes(node)
        # Add a send operator for computing KV cache transfer time
        output_node = next((n for n in reversed(self.original_prefill_FW.graph.nodes) if n.op == 'output'), None)
        if output_node is None:
            send_node = self.original_prefill_FW.graph.call_function(HcclSend, args = tuple(kv_nodes), kwargs = {})
            send_node.update_kwarg('call_module', 'PD_separation')
            send_node.update_kwarg('comm_group', [0,1])
        else:
            with self.original_prefill_FW.graph.inserting_before(output_node):
                send_node = self.original_prefill_FW.graph.call_function(HcclSend, args = tuple(kv_nodes), kwargs = {})
                send_node.update_kwarg('call_module', 'PD_separation')
                send_node.update_kwarg('comm_group', [0,1])
        # Simulate the send operation to get transfer time
        time2 += Link().get_communication_time(send_node)
        print("KV Cache Transfer Time: ", time2)

        # Step 3 : Evaluate Decode Stage
        time3 = 0.0
        time_every_step = 0.0
        for i in range(self.res_seq_len):
            curr_seq_len = self.inp_seq_len + i
            shape_env = self._get_shape_env(self.original_decode_FW)

            if shape_env is not None:
                seq_sym = self._modify_seq_sym(shape_env, curr_seq_len)
                self.decode_graph.run()
                D_Stage_Eval = Performance_Evaluation(self.decode_graph, really_run = self.really_run)
                time_every_step = D_Stage_Eval.Evaluate()
                print("Decode Step ", i+1, " Evaluation Time: ", time_every_step)
                time3 += time_every_step
                if seq_sym is not None:
                    shape_env.var_to_val[seq_sym] = self.inp_seq_len
            else:
                D_Stage_Eval = Performance_Evaluation(self.decode_graph, really_run = self.really_run)
                time_every_step = D_Stage_Eval.Evaluate()
                print("Decode Step ", i+1, " Evaluation Time: ", time_every_step)
                time3 += time_every_step

        print("Total Decode Stage Evaluation Time: ", time3)

        total_time = time1 + time2 + time3
        print("Total PD Separation Evaluation Time: ", total_time)
        return total_time
    

    def _update_graph_shapes(self, node:fx.Node):
        shape_env = self._get_shape_env(self.original_prefill_FW)
        if shape_env == None:
            return 
        # 更新 tensor_meta 中的形状
        old_shape = node.meta['tensor_meta'].shape
        new_shape = []
        flag = False
        for dim in old_shape:
            if isinstance(dim, torch.SymInt) and hasattr(dim, 'node'):
                sym_node = dim.node
                if sym_node in shape_env.var_to_val:
                    new_shape.append(shape_env.var_to_val[sym_node])
    
                else:
                    # 复合表达式（不是独立的符号） 需代入已知符号求值
                    expr = sym_node.expr
                    substituted = expr.subs({s: shape_env.var_to_val.get(s, s) for s in expr.free_symbols})
                    if substituted.is_number:
                        new_shape.append(int(substituted))
                    else:
                        new_shape.append(dim)
                flag = True
            else:
                new_shape.append(dim)
        if flag == True:
            node.meta['tensor_meta'] = node.meta['tensor_meta']._replace(shape=tuple(new_shape))
                