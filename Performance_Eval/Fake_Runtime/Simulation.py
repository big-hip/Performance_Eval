'''
    recv first, then compute, send later
    Multiprocess, single-threading version.
    Multiprocessing check data ready. Only wait.
'''
from queue import Queue, deque
import re
from numpy import double, equal
import torch
import torch.fx as fx
from collections import *
import threading
import networkx as nx
import multiprocessing as mp
# 假设 Hybrid_Parallel 在 python path 下
from Dist_IR.Hybrid_Parallel import Hybrid_Parallel 

from . import Link_manage, Memory_manage, Compute_manage
from .Link_manage import Link
from multiprocessing.managers import BaseManager
from threading import Event, local
from .Memory_manage import LifeTime
import gc, time

# ---- [关键] 强制设置启动方式为 spawn (防止重复设置报错) ----
try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass

# =============================================================================
# 1. 全局类与辅助函数 (必须在模块顶层，供 pickle 使用)
# =============================================================================

class SharedBarrier:
    def __init__(self, parties: int):
        self._barrier = mp.Barrier(parties)
    def wait(self):
        return self._barrier.wait()

class SharedManager(BaseManager): pass

# 注册共享对象
SharedManager.register("Barrier", SharedBarrier)
SharedManager.register("dict", dict)
SharedManager.register("list", list)
SharedManager.register("Lock", mp.Lock)
SharedManager.register("Value", mp.Value)
SharedManager.register("Array", mp.Array)

# --- [关键新增] 1. 模拟 TensorMetadata ---
class FakeTensorMeta:
    """
    模拟 torch.fx.passes.shape_prop.TensorMetadata
    只保存 shape 和 dtype，确保可序列化且属性可访问。
    """
    def __init__(self, shape, dtype):
        self.shape = shape
        self.dtype = dtype
    
    def __repr__(self):
        return f"FakeMeta(shape={self.shape}, dtype={self.dtype})"

# --- [关键新增] 2. TaskNode: 纯 Python 轻量级节点 ---
class TaskNode:
    """
    一个纯 Python 的轻量级节点，用于替代 torch.fx.Node 传入子进程。
    彻底解决 PyCapsule 无法被 pickle 的问题，同时携带 CostModel 所需的 meta 信息。
    """
    def __init__(self, name, op, target_name):
        self.name = name                 # 对应 node.name
        self.op = op                     # 对应 node.op
        self.target = target_name        # 对应 node.target (字符串形式)
        self.args = ()                   # 对应 node.args
        self.kwargs = {}                 # 对应 node.kwargs
        self.all_input_nodes = []        # 对应 node.all_input_nodes
        self.users = {}                  # 对应 node.users
        self.meta = {}                   # 对应 node.meta (包含 tensor_meta)

    def __repr__(self):
        return f"TaskNode(name={self.name}, op={self.op}, target={self.target})"

# --- [关键新增] 3. 转换辅助函数 ---
def _extract_meta(fx_node):
    """从真实 fx.Node 提取并转换为 FakeTensorMeta"""
    if not hasattr(fx_node, 'meta'):
        return {}
    
    new_meta = {}
    # 提取 tensor_meta
    tm = fx_node.meta.get('tensor_meta')
    if tm is not None:
        shape = getattr(tm, 'shape', None)
        dtype = getattr(tm, 'dtype', None)
        
        # 处理某些情况 tm 直接是 Tensor (如 'val')
        if shape is None and isinstance(tm, torch.Tensor):
            shape = tm.shape
            dtype = tm.dtype
        
        if shape is not None:
            # 转为 list 确保 pickle 安全
            new_meta['tensor_meta'] = FakeTensorMeta(list(shape), dtype)
    
    # 提取 val (部分 CostModel 可能依赖)
    val = fx_node.meta.get('val')
    if isinstance(val, torch.Tensor) and 'tensor_meta' not in new_meta:
        new_meta['tensor_meta'] = FakeTensorMeta(list(val.shape), val.dtype)

    return new_meta

def _map_aggregate(arg, node_map, dependency_list):
    """
    递归处理参数，将 fx.Node 替换为 TaskNode，并收集依赖。
    """
    if isinstance(arg, fx.Node):
        if arg in node_map:
            task_node = node_map[arg]
            dependency_list.append(task_node) # 记录依赖
            return task_node
        else:
            return str(arg)
            
    elif isinstance(arg, (list, tuple)):
        return type(arg)(_map_aggregate(elem, node_map, dependency_list) for elem in arg)
        
    elif isinstance(arg, dict):
        return {k: _map_aggregate(v, node_map, dependency_list) for k, v in arg.items()}
        
    elif isinstance(arg, slice):
        return slice(
            _map_aggregate(arg.start, node_map, dependency_list),
            _map_aggregate(arg.stop, node_map, dependency_list),
            _map_aggregate(arg.step, node_map, dependency_list)
        )
        
    else:
        return arg
        

def convert_fx_to_task_queue(fx_queue):
    """
    将包含 fx.Node 的队列转换为包含 TaskNode 的队列。
    """
    node_map = {}
    new_queue = deque()
    
    # Pass 1: 创建 TaskNode 骨架并提取 Meta
    for item in fx_queue:
        if isinstance(item, str): 
            new_queue.append(item)
            continue
            
        target_name = str(item.target)
        if hasattr(item.target, '__name__'):
            target_name = item.target.__name__
            
        task_node = TaskNode(item.name, item.op, target_name)
        # 提取 Meta 信息供 CostModel 使用
        task_node.meta = _extract_meta(item)
        
        node_map[item] = task_node
        new_queue.append(task_node)
        
    # Pass 2: 填充 args 和 users
    for fx_node in fx_queue:
        if isinstance(fx_node, str): continue
        
        task_node = node_map[fx_node]
        dependencies = [] 
        
        task_node.args = _map_aggregate(fx_node.args, node_map, dependencies)
        task_node.kwargs = _map_aggregate(fx_node.kwargs, node_map, dependencies)
        
        seen = set()
        for dep in dependencies:
            if dep not in seen:
                task_node.all_input_nodes.append(dep)
                seen.add(dep)
        
        for user_node in fx_node.users.keys():
            if user_node in node_map:
                user_task = node_map[user_node]
                task_node.users[user_task] = None
                
    return new_queue

# =============================================================================
# 2. 设备进程函数 (必须在模块顶层)
# =============================================================================

def device_process(device_id:int, 
                   sched_queue, 
                   local_time_list, 
                   link_manage:list[Link], 
                   group_barriers, 
                   pp_stage_gap, 
                   global_lock,
                   really_run) -> double:
    
    print(f"[Device {device_id}] Process started (PID: {mp.current_process().pid}).")
    group_barriers[f'All-devices-sync'].wait()

    local_time = [0.0]
    compute_q = Queue()
    comm_q = Queue()
    mem_dict = {}
   
    scheduler = DeviceScheduler()
    # 注意：link_manage 传进来是 list，取第一个
    link_manage_inst = link_manage[0]

    # --- 预扫描队列，分离计算与通信任务 ---
    recv_placeholder_registered = False
    for task in sched_queue:
        if isinstance(task, str): continue 

        if task.op == 'placeholder':
            if f'hccl_recv_' not in task.name:
                tensor_life = Memory_manage.LifeTime()
                tensor_life.start_time = local_time[0]
                mem_dict[task.name] = tensor_life
            else:
                tensor_life = Memory_manage.LifeTime()
                tensor_life.wait = True
                mem_dict[task.name] = tensor_life
                if not recv_placeholder_registered:
                    comm_q.put('Recv_placeholders')
                    recv_placeholder_registered = True
        
        elif task.op == 'call_function':
            task_target_name = str(task.target)
            if 'hccl' in task_target_name.lower():
                comm_q.put(task)
            else:
                compute_q.put(task)
            
        elif task.op == 'output':
            mem_dict[task.name] = task

    if recv_placeholder_registered:
        sched_queue.appendleft('Recv_placeholders')
    
    # 转换为线程安全的 Queue
    q = Queue()
    for item in sched_queue:
        q.put(item)

    from collections import defaultdict
    barrier_rounds = defaultdict(int)
    compute_manage = Compute_manage.Compute(q, really_run=really_run)
    
    # --- 主循环 ---
    group_barriers[f"All-devices-sync"].wait()
    while True:
        if q.empty():
            break

        task_node = q.get()
        is_node = hasattr(task_node, 'op')
        
        target = ""
        if isinstance(task_node, str):
            target = task_node
        elif is_node:
            target = str(task_node.target).lower()

        if 'hccl' in target or task_node == 'Recv_placeholders':
            # 检查通信依赖
            if task_node == 'Recv_placeholders' or link_manage_inst.check_comm_data_ready(task_node, mem_dict):   
                
                # --- CASE 1: Recv Placeholder ---
                if task_node == 'Recv_placeholders':
                    comm_group = [max(0, device_id - pp_stage_gap), device_id]
                    call_module = 'PP'
                    barrier_key = f"{call_module}-send/recv-" + "-".join(str(x) for x in sorted(comm_group))

                    if barrier_key in group_barriers:
                        barrier = group_barriers[barrier_key]
                        barrier_rounds[barrier_key] += 1
                        
                        scheduler.begin_comm(overlap=False)
                        try:
                            # 显式使用锁
                            with global_lock:
                                comm_time = link_manage_inst.get_communication_time("Recv_placeholders") 
                                local_time[0] += comm_time
                                local_time_list[device_id] = local_time[0]

                            barrier.wait() 

                            with global_lock:
                                group_times = [local_time_list[r] if not isinstance(local_time_list[r], list) else local_time_list[r][0] for r in comm_group]
                                group_max_time = max(group_times)
                                for r in comm_group:
                                    if local_time_list[r] != group_max_time:
                                        local_time_list[r] = group_max_time
                                local_time[0] = local_time_list[device_id]

                                for name, life in mem_dict.items():
                                    if isinstance(life, Memory_manage.LifeTime) and life.wait:
                                        life.wait = False
                                        life.start_time = local_time[0]

                                tensor_life = Memory_manage.LifeTime()
                                tensor_life.start_time = local_time[0]
                                mem_dict[f"recv_complete_{device_id}"] = tensor_life 
                            
                            barrier.wait() 

                        except Exception as e:
                            print(f"[ERROR][Device {device_id}] Recv_placeholders failed: {e}")
                        finally:
                            scheduler.end_comm()
                    else:
                        pass 

                    continue

                # --- CASE 2: Normal Communication ---
                comm_type = str(task_node.target)
                if "send" in comm_type.lower(): comm_type = "send/recv"
                
                comm_group = task_node.kwargs.get("comm_group", [device_id])
                overlap = task_node.kwargs.get("overlap", False)
                call_module = task_node.kwargs.get("call_module", "Unknown")

                group_key = f"{call_module}-{comm_type}-" + "-".join(str(int(x)) for x in sorted(comm_group))
                
                if group_key in group_barriers:
                    barrier = group_barriers[group_key]
                    barrier_rounds[group_key] += 1
                    
                    if "send" in comm_type: # P2P Send
                        scheduler.begin_comm(overlap=overlap)
                        try:
                            with global_lock:
                                comm_time = link_manage_inst.get_communication_time(task_node)
                                local_time[0] += comm_time
                                local_time_list[device_id] = local_time[0]
                            
                            barrier.wait()

                            with global_lock:
                                group_times = [local_time_list[r] if not isinstance(local_time_list[r], list) else local_time_list[r][0] for r in comm_group]
                                group_max_time = max(group_times)
                                for r in comm_group:
                                    if local_time_list[r] != group_max_time:
                                        local_time_list[r] = group_max_time
                                local_time[0] = local_time_list[device_id]
                                
                                tensor_life = Memory_manage.LifeTime()
                                tensor_life.start_time = local_time[0]
                                mem_dict[task_node.name] = tensor_life

                                if task_node.all_input_nodes:
                                    for inp in task_node.all_input_nodes:
                                        can_release = True
                                        for succ in list(inp.users.keys()):
                                            if succ.name not in mem_dict:
                                                can_release = False
                                            elif isinstance(mem_dict[succ.name], Memory_manage.LifeTime) and mem_dict[succ.name].wait:
                                                can_release = False
                                        if can_release and inp.name in mem_dict:
                                            mem_dict[inp.name].end_time = local_time[0]
                            barrier.wait()
                        except Exception as e:
                            print(f"[ERROR][Device {device_id}] Send failed: {e}")
                        finally:
                            scheduler.end_comm()

                    else: # Collective
                        scheduler.begin_comm(overlap=False)
                        try:
                            with global_lock:
                                comm_time = link_manage_inst.get_communication_time(task_node)
                                local_time[0] += comm_time
                                local_time_list[device_id] = local_time[0]
                            
                            barrier.wait()

                            with global_lock:
                                group_times = [local_time_list[r] if not isinstance(local_time_list[r], list) else local_time_list[r][0] for r in comm_group]
                                group_max_time = max(group_times)
                                for r in comm_group:
                                    if local_time_list[r] != group_max_time:
                                        local_time_list[r] = group_max_time
                                local_time[0] = local_time_list[device_id]
                                
                                tensor_life = Memory_manage.LifeTime()
                                tensor_life.start_time = local_time[0]
                                mem_dict[task_node.name] = tensor_life

                                if task_node.all_input_nodes:
                                    for inp in task_node.all_input_nodes:
                                        can_release = True
                                        for succ in list(inp.users.keys()):
                                            if succ.name not in mem_dict:
                                                can_release = False
                                            elif isinstance(mem_dict[succ.name], Memory_manage.LifeTime) and mem_dict[succ.name].wait:
                                                can_release = False
                                        if can_release and inp.name in mem_dict:
                                            mem_dict[inp.name].end_time = local_time[0]
                            barrier.wait()
                        except Exception as e:
                            print(f"[ERROR][Device {device_id}] Collective failed: {e}")
                        finally:
                            scheduler.end_comm()
                    continue

        elif is_node and task_node.op == 'call_function':
            while True:
                with global_lock:
                    if scheduler.can_compute() and compute_manage.check_compute_ready(task_node, mem_dict):
                        scheduler.begin_compute()
                        comp_time = compute_manage.get_computation_time(task_node)
                        local_time[0] += comp_time
                        local_time_list[device_id] = local_time[0]

                        tensor_life = Memory_manage.LifeTime()
                        tensor_life.start_time = local_time[0]
                        mem_dict[task_node.name] = tensor_life

                        if task_node.all_input_nodes:
                            for inp in task_node.all_input_nodes:
                                can_release = True
                                for succ in list(inp.users.keys()):
                                    if succ.name not in mem_dict:
                                        can_release = False
                                    elif isinstance(mem_dict[succ.name], Memory_manage.LifeTime) and mem_dict[succ.name].wait:
                                        can_release = False
                                if can_release and inp.name in mem_dict:
                                    mem_dict[inp.name].end_time = local_time[0]
            
                        scheduler.end_compute()
                        break
                    else:
                        pass
                time.sleep(0.01) 

        else:                
            continue

    group_barriers[f"All-devices-sync"].wait()
    return local_time[0]


# =============================================================================
# 3. 调度与评估类
# =============================================================================

class DeviceScheduler:
    def __init__(self):
        self.lock = threading.Lock()
        self.compute_busy = False
        self.comm_busy = False

    def can_compute(self):
        with self.lock:
            state = not self.comm_busy
        return state

    def begin_compute(self):
        with self.lock: self.compute_busy = True

    def end_compute(self):
        with self.lock: self.compute_busy = False

    def begin_comm(self, overlap=False):
        with self.lock:
            if not overlap: self.comm_busy = True

    def end_comm(self):
        with self.lock: self.comm_busy = False


def Topological_sort(rank, Graphmodule: fx.GraphModule) -> deque:
    placeholder_nodes = []
    call_nodes = []
    output_nodes = []
    for node in Graphmodule.graph.nodes:
        if node.op == "placeholder": placeholder_nodes.append(node)
        elif node.op in ["call_function", "call_module"]: call_nodes.append(node)
        elif node.op == "output": output_nodes.append(node)
        else: call_nodes.append(node)
    return deque(placeholder_nodes + call_nodes + output_nodes)


class InferenceSchedule:
    def __init__(self, DistIR, KVcache:bool) -> None:
        self.distir = DistIR
        self.kvcache = KVcache
        if KVcache == True:
            self.schedule_queue = self._schedule_with_kvcache(DistIR)
        else:
            self.schedule_queue = self._schedule(DistIR)
    
    def _schedule(self, DistIR) -> list[deque]:
        schedule_queue = []
        rank =  0
        for d in DistIR:
            for p in d:
                for t in p:
                    for e in t:
                        for s in e:
                            FW_graph = s['FW']
                            sorted_q = Topological_sort(rank, FW_graph)
                            schedule_queue.append(sorted_q)
                            rank += 1
        return schedule_queue

    def _schedule_with_kvcache(self, DistIR) -> list[deque]:
        return self._schedule(DistIR)

    def schedule(self) -> list[deque]:
        return self.schedule_queue
    

class Performance_Evaluation:
    def __init__(self, HybridParallel:Hybrid_Parallel, Topology_Graph:nx.Graph = nx.Graph(), TaskGraph:nx.Graph = nx.Graph(), Mode:str ='Inference', KV_Cache:bool = False, PP_sched:str = 'FthenB', really_run:bool = False) -> None: 
        self.Dist_IR = HybridParallel.dist_IR
        self.Device_nums = HybridParallel.device_nums
        self.T = HybridParallel.T
        self.M = HybridParallel.M
        self.S = HybridParallel.S
        self.Topology_Graph = Topology_Graph
        self.TaskGraph = TaskGraph
        self.Mode = Mode
        self.KV_Cache = KV_Cache
        self.PP_sched = PP_sched
        self.really_run = really_run

    def Evaluate(self) -> double:
        assert self.Mode in ['Training','Inference']
        if self.Mode == 'Training':
            return self._Training_schedule(self.Dist_IR, self.Topology_Graph, self.TaskGraph, self.PP_sched)
        elif self.Mode == 'Inference':
            return self._Inference_schedule(self.Dist_IR, self.Topology_Graph, self.KV_Cache)
    
    def _Training_schedule(self, Dist_IR, Topology_Graph, TaskGraph, PP_sched) -> double:
        return 0.0
    
    def _Inference_schedule(self, Dist_IR: list[fx.GraphModule], Topology_Graph: nx.Graph, KV_Cache: bool) -> double:
        print("[Main] Initializing shared resources (Spawn Mode)...")
        
        # 1. 获取 Spawn Context
        ctx = mp.get_context('spawn')
        
        link_manage = Link(Topology_Graph, self.really_run)
        pp_stage_gap = self.T * self.M * self.S 

        # 2. 提取通信组
        all_comm_groups = set()
        for d in Dist_IR:
            for p in d:
                for t in p:
                    for e in t:
                        for s in e:
                            FW_graph = s["FW"]
                            for node in FW_graph.graph.nodes:
                                if node.op == "call_function" and "hccl" in str(node.target).lower():
                                    comm_group = tuple(sorted(int(x) for x in node.kwargs.get("comm_group", [])))
                                    call_module = node.kwargs.get("call_module", "Unknown")
                                    comm_type = str(node.target.__name__) if hasattr(node.target, '__name__') else str(node.target)
                                    if "send" in comm_type.lower(): comm_type = "send/recv"
                                    if len(comm_group) >= 1:
                                        barrier_key = f"{call_module}-{comm_type}-" + "-".join(str(x) for x in comm_group)
                                        all_comm_groups.add((barrier_key, len(comm_group)))

        # 3. 启动 Manager
        manager = SharedManager()
        manager.start()
        
        local_time_list = ctx.Array('d', [0.0] * self.Device_nums)
        # --- [修复] 使用 ctx.Lock() 代替 manager.Lock() ---
        global_lock = ctx.Lock()
        
        global_barriers = {}

        for key, group_size in all_comm_groups:
            if key not in global_barriers:
                global_barriers[key] = manager.Barrier(group_size)

        # 4. 生成调度队列并转换为 TaskNode
        Infer_sched = InferenceSchedule(Dist_IR, KV_Cache)
        raw_sched_queue = Infer_sched.schedule()
        real_device_nums = len(raw_sched_queue)
        global_barriers["All-devices-sync"] = manager.Barrier(real_device_nums)

        print("[Main] Converting fx.Graph to multiprocessing-safe TaskGraph...")
        safe_sched_queue = []
        for q in raw_sched_queue:
            safe_sched_queue.append(convert_fx_to_task_queue(q))
        # def debug_print_task_queue(queue, device_id, limit=5):
        #     print(f"\n{'='*20} [Debug] Inspecting Queue for Device {device_id} {'='*20}")
            
        #     count = 0
        #     for item in queue:
        #         if count >= limit:
        #             print("... (remaining nodes omitted) ...")
        #             break
                    
        #         if isinstance(item, str):
        #             print(f"[{count}] [MARKER] {item}")
        #         else:
        #             print(f"[{count}] Node: {item.name}")
        #             print(f"    Op:     {item.op}")
        #             print(f"    Target: {item.target}") # 注意这里现在是字符串
                    
        #             # --- 新增：打印参数，证明拓扑结构还在 ---
        #             if item.args:
        #                 # 简略打印 args，只显示引用的 Node 名字
        #                 readable_args = []
        #                 for a in item.args:
        #                     if hasattr(a, 'name'): readable_args.append(f"Node({a.name})")
        #                     else: readable_args.append(str(a))
        #                 print(f"    Args:   {readable_args}")

        #             # --- 新增：打印 Kwargs，证明通信组还在 ---
        #             if item.kwargs:
        #                 print(f"    Kwargs: {item.kwargs}")

        #             # 打印 Meta
        #             if hasattr(item, 'meta') and 'tensor_meta' in item.meta:
        #                 tm = item.meta['tensor_meta']
        #                 print(f"    Meta:   Shape={tm.shape}, Dtype={tm.dtype}")
        #             else:
        #                 print(f"    Meta:   [MISSING]")

        #             # 打印依赖
        #             input_names = [n.name for n in item.all_input_nodes]
        #             print(f"    Inputs: {input_names}")

        #         print("-" * 40)
        #         count += 1
        #     print(f"{'='*60}\n")
        # if len(safe_sched_queue) > 0:
        #     print("[Main] Debugging Device 0 Queue structure:")
        #     debug_print_task_queue(safe_sched_queue[0], device_id=0, limit=1000)
        # =======================
        # 5. 启动子进程
        processes = []
        for i in range(real_device_nums):
            p = ctx.Process(
                target=device_process,
                args=(i, safe_sched_queue[i], local_time_list, [link_manage], global_barriers, pp_stage_gap, global_lock, self.really_run),
                daemon=True
            )
            processes.append(p)
        
        print("[Main] Starting all device processes...")
        for p in processes: p.start()
        
        for p in processes: p.join()

        with global_lock:
            Global_time = round(max(local_time_list[:]), 2)
        
        try:
            manager.shutdown()
        except: pass
        
        gc.collect()
        return Global_time
        # return Global_time ,local_time_list