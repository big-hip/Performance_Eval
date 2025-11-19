'''
    Recv first then compute, send last.
    Multiprocess + Multithreading version.
    Multiprocessing check data ready. Only wait.
'''
from queue import Queue
import re
from numpy import double, equal
import torch.fx as fx
from collections import *
import threading
import networkx as nx
import multiprocessing as mp
from Dist_IR.Hybrid_Parallel import Hybrid_Parallel 

from . import Link_manage, Memory_manage, Compute_manage
from .Link_manage import Link
from multiprocessing.managers import BaseManager
from threading import Event, local
from .Memory_manage import LifeTime
mp.set_start_method("fork", force=True)
import gc, time

# ---- 可共享 Barrier 定义 ----
class SharedBarrier:
    def __init__(self, parties: int):
        self._barrier = mp.Barrier(parties)
    def wait(self):
        return self._barrier.wait()

# ---- 自定义 Manager 注册共享对象 ----
class SharedManager(BaseManager): pass

SharedManager.register("Barrier", SharedBarrier)
SharedManager.register("dict", dict)
SharedManager.register("list", list)
SharedManager.register("Lock", mp.Lock)
SharedManager.register("Value", mp.Value)
SharedManager.register("Array", mp.Array)

"""
    The specific approach is as follows:
        First, depending on whether it's an inference scenario or a training scenario, 
        schedule DistIR into a queue to be executed by n devices. 
        Then, have each process simulate one device executing this queue. 
        Next, use three threads to simulate the computation, communication, and memory management during execution.
        Evaluate the performance of these operations using the cost model. 
        Finally, obtain a complete end-to-end performance evaluation time.

"""



class DeviceScheduler:
    """
        Device-level scheduler to coordinate overlap between computation and communication.
    """
    def __init__(self):
        self.lock = threading.Lock()
        self.compute_busy = False
        self.comm_busy = False
        print("[DEBUG][DeviceScheduler.__init__] Initialized scheduler with lock.")

    def can_compute(self):
        with self.lock:
            state = not self.comm_busy
        # print(f"[DEBUG][DeviceScheduler.can_compute] can_compute={state}")
        return state

    def begin_compute(self):
        with self.lock:
            self.compute_busy = True
        print("[DEBUG][DeviceScheduler.begin_compute] Computation started.")

    def end_compute(self):
        with self.lock:
            self.compute_busy = False
        print("[DEBUG][DeviceScheduler.end_compute] Computation ended.")

    def begin_comm(self, overlap=False):
        with self.lock:
            if not overlap:
                self.comm_busy = True
        print(f"[DEBUG][DeviceScheduler.begin_comm] Communication started. overlap={overlap}")

    def end_comm(self):
        with self.lock:
            self.comm_busy = False
        print("[DEBUG][DeviceScheduler.end_comm] Communication ended.")



def Topological_sort(rank, Graphmodule: fx.GraphModule) -> deque:
    """
        Ensure that fx.Graph satisfies topological execution order:
        1. Placeholders first
        2. Call_function / Call_module next
        3. Output nodes last
    """
    # print(f"[DEBUG][Topological_sort] Sorting graph: {Graphmodule}")

    placeholder_nodes = []
    call_nodes = []
    output_nodes = []

    # 分类节点
    for node in Graphmodule.graph.nodes:
        if node.op == "placeholder":
            placeholder_nodes.append(node)
        elif node.op in ["call_function", "call_module"]:
            call_nodes.append(node)
        elif node.op == "output":
            output_nodes.append(node)
        else:
            print(f"[WARN][Topological_sort] Unknown node type {node.op} for {node.name}, appending to call_nodes.")
            call_nodes.append(node)

    # 拼接成执行顺序
    ordered_nodes = placeholder_nodes + call_nodes + output_nodes

    task_queue = deque(ordered_nodes)

    # Debug 信息
    print(f"[DEBUG {rank}][Topological_sort] Added {len(placeholder_nodes)} placeholders, "
          f"{len(call_nodes)} call nodes, {len(output_nodes)} outputs.")
    print(f"[DEBUG {rank}][Topological_sort] Total nodes sorted: {len(task_queue)}")

    return task_queue




class TrainSchedule:
    """
        The task graph after PP policy scheduling must be passed in.
        This task graph contains information about microbatches and stages.

    """
    def __init__(self, Dist_IR:list[fx.GraphModule], TaskGraph:nx.Graph, PP_sched:str) -> None:
        self.distir = Dist_IR
        self.taskgraph = TaskGraph
        self.PP_sched = PP_sched
        print(f"[DEBUG][TrainSchedule.__init__] Initialized with PP_sched={PP_sched}, Graph nodes={len(TaskGraph.nodes)}")
        #TODO
        
    def Schedule(self):
        print("[DEBUG][TrainSchedule.Schedule] Not implemented yet.")
        pass


class InferenceSchedule:
    """
        Handle two inference scenarios: one with KV Cache optimization and one without.
        Uni_IR = dict{'FW':GraphModule,'BW':GraphModule,'OPT':GraphModule}
    """
    def __init__(self, DistIR, KVcache:bool) -> None:
        self.distir = DistIR
        self.kvcache = KVcache
        print(f"[DEBUG][InferenceSchedule.__init__] Initializing inference schedule with KVcache={KVcache}")
        if KVcache == True:
            self.schedule_queue = self._schedule_with_kvcache(DistIR)
        else:
            self.schedule_queue = self._schedule(DistIR)
    

    def _schedule(self, DistIR) -> list[deque]:
        schedule_queue = []
        print("[DEBUG][InferenceSchedule._schedule] Generating non-KVCache schedule.")
        rank =  0
        for d in DistIR:
            for p in d:
                for t in p:
                    for s in t:
                        FW_graph = s['FW']
                        # print(f"[DEBUG][InferenceSchedule._schedule] Processing FW graph {FW_graph}")
                        sorted = Topological_sort(rank, FW_graph)
                        schedule_queue.append(sorted)
                        rank += 1
        print(f"[DEBUG][InferenceSchedule._schedule] Generated {len(schedule_queue)} schedule queues.")
        return schedule_queue


    def _schedule_with_kvcache(self, DistIR) -> list[deque]:
        #TODO
        schedule_queue = []
        print("[DEBUG][InferenceSchedule._schedule_with_kvcache] Generating KVCache schedule.")
        for d in DistIR:
            for p in d:
                for t in p:
                    for s in t:
                        FW_graph = s['FW']
                        # print(f"[DEBUG][InferenceSchedule._schedule_with_kvcache] Processing FW graph {FW_graph}")
                        sorted = Topological_sort(FW_graph)
                        schedule_queue.append(sorted)
        print(f"[DEBUG][InferenceSchedule._schedule_with_kvcache] Generated {len(schedule_queue)} schedule queues.")
        return schedule_queue

    def schedule(self) -> list[deque]:
        print(f"[DEBUG][InferenceSchedule.schedule] Returning {len(self.schedule_queue)} prepared queues.")
        return self.schedule_queue
    
    


class Performance_Evaluation:
    """
        Perform performance evalution to output end-to-end performance metrics for training or inference.
        args:
                Dist_IR: Distributed Computational Graph Based on Fx.GraphModule.
                Topology_Gaph: Moding devive topology graph by myself.
                Mode: Optional Modes,Training or Inferernce.
    """
    def __init__(self, HybridParallel:Hybrid_Parallel, Topology_Graph:nx.Graph = nx.Graph(), TaskGraph:nx.Graph = nx.Graph(), Mode:str ='Inference', KV_Cache:bool = False, PP_sched:str = 'FthenB') -> None: 
        self.Dist_IR = HybridParallel.dist_IR
        self.Device_nums = HybridParallel.device_nums
        self.D = HybridParallel.D
        self.P = HybridParallel.P
        self.T = HybridParallel.T
        self.E = HybridParallel.E
        self.S = HybridParallel.S

        self.Topology_Graph = Topology_Graph
        self.TaskGraph = TaskGraph
        self.Mode = Mode
        self.KV_Cache = KV_Cache
        self.PP_sched = PP_sched
        print(f"[DEBUG][Performance_Evaluation.__init__] Initialized with Mode={Mode}, Devices={self.Device_nums}")


    def Evaluate(self) -> double:
        print(f"[DEBUG][Performance_Evaluation.Evaluate] Evaluating in mode {self.Mode}")
        assert self.Mode in ['Training','Inference'],'The mode can only be selected from [Inference, Training]'
        if self.Mode not in ['Training', 'Inference']:
            raise RuntimeError('Incorrect mode selection,only Training or Inference!')
        
        elif self.Mode == 'Training':
            print("[DEBUG][Performance_Evaluation.Evaluate] Entering training schedule.")
            return self._Training_schedule(self.Dist_IR, self.Topology_Graph, self.TaskGraph, self.PP_sched)
        
        elif self.Mode == 'Inference':
            print("[DEBUG][Performance_Evaluation.Evaluate] Entering inference schedule.")
            return self._Inference_schedule(self.Dist_IR, self.Topology_Graph, self.KV_Cache)
    

    def _Training_schedule(self, Dist_IR:list[fx.GraphModule], Topology_Graph:nx.Graph, TaskGraph:nx.Graph, PP_sched:str) -> double:
        Global_time = 0
        print("[DEBUG][Performance_Evaluation._Training_schedule] Training schedule placeholder.")
        return Global_time
    

    def _Inference_schedule(self, Dist_IR: list[fx.GraphModule], Topology_Graph: nx.Graph, KV_Cache: bool) -> double:
        print("[Main] Initializing shared resources and launching device processes...")
        link_manage = Link(Topology_Graph)
        # 相邻PP-stage间隔设备数
        pp_stage_gap = self.T * self.S 

        # 用 multiprocessing.Manager 共享时间
        manager = mp.Manager()
        local_time_list = mp.Array('d', [0.0] * self.Device_nums)
        global_lock = manager.Lock() 

        # Step 1. 提取所有通信组 (scan all comm groups, with comm domain)
        all_comm_groups = set()
        for d in Dist_IR:
            for p in d:
                for t in p:
                    for s in t:
                        FW_graph = s["FW"]
                        for node in FW_graph.graph.nodes:
                            if node.op == "call_function" and "hccl" in str(node.target).lower():
                                comm_group = tuple(sorted(int(x) for x in node.kwargs.get("comm_group", [])))
                                call_module = node.kwargs.get("call_module", "Unknown")
                                comm_type = str(node.target.__name__) if hasattr(node.target, '__name__') else str(node.target)
                                if "send" in comm_type.lower():
                                    comm_type = "send/recv"
                                if len(comm_group) >= 1:
                                    barrier_key = f"{call_module}-{comm_type}-" + "-".join(str(x) for x in comm_group)
                                    all_comm_groups.add((barrier_key, len(comm_group)))
        print(f"[Main] Found {len(all_comm_groups)} unique communication groups (with comm domains).")

        # Step 2. 使用 SharedManager 创建可共享 Barrier
        manager = SharedManager()
        manager.start()
        global_barriers = {}

        for key, group_size in all_comm_groups:
            if key not in global_barriers:
                global_barriers[key] = manager.Barrier(group_size)
                print(f"[Main] Created shared barrier for group={key} (size={group_size})")



        # Step 4. 生成推理调度队列
        Infer_sched = InferenceSchedule(Dist_IR, KV_Cache)
        sched_queue = Infer_sched.schedule()
        real_device_nums = len(sched_queue)
        print(f"[Main] Prepared {real_device_nums} device queues for inference scheduling.")

        # 确保全局 barrier 设备数匹配
        global_barriers["All-devices-sync"] = manager.Barrier(real_device_nums)

        # Step 5. 启动子进程
        processes = []
        for i in range(real_device_nums):
            p = mp.Process(
                target=device_process,
                args=(i, sched_queue[i], local_time_list, [link_manage], global_barriers, pp_stage_gap, global_lock),
                daemon=True
            )
            processes.append(p)

        print("[Main] Starting all device processes...")
        for p in processes:
            p.start()
        for p in processes:
            p.join()

        # Step 6. 汇总全局时间
        with global_lock:
            Global_time = round(max(local_time_list[:]), 2)
        print(f"[Main] All devices finished. Global time = {Global_time}")

        #  Step 7. fork 模式下主动关闭 Manager，防止多轮 Evaluate() 残留
        try:
            if isinstance(manager, SharedManager):
                manager.shutdown()
                print("[Main] SharedManager shutdown completed.")
        except Exception as e:
            print(f"[WARN] Failed to shutdown SharedManager cleanly: {e}")

        #  Step 8. 回收内存资源
        import gc, time
        gc.collect()
        time.sleep(0.2)

        return Global_time

# ----------------------------
# --- Computation Thread ---
# ----------------------------
def compute_thread(device_id:int, 
                   compute_q:Queue, 
                   local_time:list[double], 
                   mem_dict:dict, 
                   local_time_list:dict[double], 
                   scheduler:DeviceScheduler, 
                   lock,
                   global_lock,
                   recv_first,
                   send_last) -> None:
    
    compute_manage = Compute_manage.Compute(compute_q)
    print(f"[Device {device_id}] Compute thread started.")
    while True:
        if not compute_q.empty():
            task_node  = compute_q.get()
            print(f"[Device {device_id}] Compute node: {task_node.name}")

            while True:
                # while not scheduler.can_compute():
                #     time.sleep(2)
                if not recv_first.is_set():
                    print(f"[Device {device_id}] Waiting for Recv...")
                    recv_first.wait()  # 一次性阻塞直到通信线程 set()
                print(f"[Device {device_id}] Recv completed, proceeding to compute {task_node.name}.")
                if recv_first.is_set():
                    with lock:
                    # if recv_first[0] == True:
                        with global_lock:
                            if scheduler.can_compute() and compute_manage.check_compute_ready(task_node, mem_dict):
                                scheduler.begin_compute()
                                comp_time = compute_manage.get_computation_time(task_node)
                                print(f"[DEBUG][Device {device_id}] Computation time for {task_node.name}: {comp_time}")
                                local_time[0] += comp_time
                                local_time_list[device_id] = local_time[0]

                                print(f"[Device {device_id}] Finished computation of {task_node.name}, local_time={local_time[0]}")
                    
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
                                        if can_release:
                                            mem_dict[inp.name].end_time = local_time[0]
                                            print(f"[DEBUG][Device {device_id}] Released memory for {inp.name} at time {local_time[0]}")
                    
                                scheduler.end_compute()
                                break
                            else:
                                pass
                else:
                    print(f"[Device {device_id}] Waiting for recv to complete before computing {task_node.name}... recv_first={recv_first}")  
                    time.sleep(0.001)
        else:
            break

    print(f"[Device {device_id}] Compute thread finished.")
    send_last.set()
    return None



# ----------------------------
# --- Communication Thread ---
# ----------------------------
def comm_thread(device_id: int,
                comm_q: Queue,
                local_time: list[double],
                mem_dict: dict,
                local_time_list: dict,
                link_manager: list[Link],
                group_barriers: dict,
                scheduler: DeviceScheduler,
                pp_stage_gap:int,
                lock,
                global_lock,
                recv_first,
                send_last) -> None:

    """
        Thread that handles communication tasks (Send/Recv/Collective).
        Each communication is modeled as time consumption + memory lifecycle updates.
        Uses Barrier synchronization across all participating devices (including single-device).
        ---- Enhanced Version ----
        Add communication domain awareness (DP/TP/PP/EP) by including node.kwargs['call_module'] in barrier keys.
    """
    link_manage: Link = link_manager[0]
    print(f"[Device {device_id}] Communication thread started.")

    from collections import defaultdict
    barrier_rounds = defaultdict(int)  # For debug tracking barrier reuse rounds

    while True:
        if comm_q.empty():
            break

        task_node = comm_q.get()
        while True:
            with lock:
                if task_node == 'Recv_placeholders' or link_manage.check_comm_data_ready(task_node, mem_dict):  
                
                    # ===========================================================
                    # ================ Recv Placeholder =========================
                    # ===========================================================
                    if task_node == 'Recv_placeholders':
                        print(f"[Device {device_id}] Handling Recv_placeholders task.")
                        comm_group = [max(0, device_id - pp_stage_gap), device_id]
                        # ---- 默认属于 Pipeline 域 ----
                        call_module = 'PP'
                        barrier_key = f"{call_module}-send/recv-" + "-".join(str(x) for x in sorted(comm_group))

                        if barrier_key not in group_barriers:
                            print(f"[WARN][Device {device_id}] Creating temporary barrier for {barrier_key}")
                            group_barriers[barrier_key] = mp.Barrier(len(comm_group))

                        barrier = group_barriers[barrier_key]
                    
                        round_id = barrier_rounds[barrier_key]
                        
                        scheduler.begin_comm(overlap=False)
                        try:
                            # --- Simulated communication delay ---
                            with global_lock:
                                comm_time = link_manage.get_communication_time("Recv_placeholders") 
                                local_time[0] += comm_time
                                local_time_list[device_id] = local_time[0]
                            print(f"[Device {device_id}] Recv_placeholders complete. +{comm_time}, local_time={local_time[0]}")

                            # --- Barrier Synchronization ---
                            print(f"[Device {device_id}] Waiting at Recv barrier for {barrier_key} (round={round_id})")
                            barrier.wait()
                            print(f"[Device {device_id}] Recv barrier released for {barrier_key} (round={round_id})")

                            # --- Synchronize local_time to group max ---
                            with global_lock:
                                group_times = []
                                for r in comm_group:
                                    val = local_time_list[r]
                                    if isinstance(val, list):
                                        val = val[0]
                                    group_times.append(val)
                                # 判断所有设备时间是不是相同，不相同就得同步
                                group_max_time = max(group_times)
                                for r in comm_group:
                                    if local_time_list[r] != group_max_time:
                                        local_time_list[r] = group_max_time

                                local_time[0] =  local_time_list[device_id]
                                print(f"[DEBUG][Device {device_id}] Synced local_time={local_time[0]} to max={group_max_time}")

                                # --- Activate all waiting tensors ---
                                for name, life in mem_dict.items():
                                    if isinstance(life, Memory_manage.LifeTime) and life.wait:
                                        life.wait = False
                                        life.start_time = local_time[0]

                                # --- Register received tensor ---
                                tensor_life = Memory_manage.LifeTime()
                                tensor_life.start_time = local_time[0]
                                mem_dict[f"recv_complete_{device_id}"] = tensor_life 
                            barrier.wait()    
                            # recv_first[0] = True  
                            recv_first.set()  
                            print(f"[Device {device_id}] All recv placeholders activated, recv_first={recv_first}.")
                            barrier_rounds[barrier_key] += 1
                              

                        except Exception as e:
                            print(f"[ERROR][Device {device_id}] Recv_placeholders failed: {e}")
                        finally:
                            scheduler.end_comm()

                        break

                    # ===========================================================
                    # ================ Handle True Communication Node ===========
                    # ===========================================================
                    comm_type = str(task_node.target.__name__) if hasattr(task_node.target, '__name__') else str(task_node.target)
                    if "send" in comm_type.lower():
                        comm_type = "send/recv"
                    comm_group = task_node.kwargs.get("comm_group", [device_id])
                    overlap = task_node.kwargs.get("overlap", False)
                    call_module = task_node.kwargs.get("call_module", "Unknown")  # 通信策略域

                    # ---- 构造唯一 Barrier Key ----
                    group_key = f"{call_module}-{comm_type}-" + "-".join(str(int(x)) for x in sorted(comm_group))
                    if group_key not in group_barriers:
                        print(f"[WARN][Device {device_id}] Missing barrier for {group_key}, creating temporarily.")
                        group_barriers[group_key] = mp.Barrier(len(comm_group))

                    barrier = group_barriers[group_key]
                   
                    round_id = barrier_rounds[group_key]

                    print(f"[DEBUG][Device {device_id}] Processing comm node type={comm_type}, domain={call_module}, group={comm_group}, round={round_id}")

                    # ===========================================================
                    # ================ Point-to-Point Send ======================
                    # ===========================================================
                    if "send" in comm_type:
                        if send_last.is_set():
                            scheduler.begin_comm(overlap=overlap)
                            try:
                                # --- Simulated communication delay ---
                                with global_lock:
                                    comm_time = link_manage.get_communication_time(task_node)
                                    local_time[0] += comm_time
                                    local_time_list[device_id] = local_time[0]
                                print(f"[Device {device_id}] Send complete. +{comm_time}, local_time={local_time[0]}")

                                # --- Barrier Synchronization ---
                                print(f"[Device {device_id}] Waiting at Send barrier for {group_key} (round={round_id})")
                                barrier.wait()
                                print(f"[Device {device_id}] Send barrier released for {group_key} (round={round_id})")

                                # --- Synchronize local_time to group max ---
                                with global_lock:
                                    group_times = []
                                    for r in comm_group:
                                        val = local_time_list[r]
                                        if isinstance(val, list):
                                            val = val[0]
                                        group_times.append(val)
                                    # 判断所有设备时间是不是相同，不相同就得同步
                                    group_max_time = max(group_times)
                                    for r in comm_group:
                                        if local_time_list[r] != group_max_time:
                                            local_time_list[r] = group_max_time
                                            
                                    local_time[0] =  local_time_list[device_id]
                                    print(f"[DEBUG][Device {device_id}] Synced local_time={local_time[0]} to max={group_max_time}")
                                    # --- Register tensor lifecycle ---
                                    tensor_life = Memory_manage.LifeTime()
                                    tensor_life.start_time = local_time[0]
                                    mem_dict[task_node.name] = tensor_life
                                    print(f"[DEBUG][Device {device_id}] Registered send tensor {task_node.name} at time {local_time[0]}")

                                    # --- Check input tensors for release ---
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
                                                print(f"[DEBUG][Device {device_id}] Released memory after send for {inp.name} at {local_time[0]}")
                                barrier.wait()
                                barrier_rounds[group_key] += 1

                            except threading.BrokenBarrierError:
                                print(f"[WARN][Device {device_id}] Barrier timeout for {group_key}")
                            except Exception as e:
                                print(f"[ERROR][Device {device_id}] Send failed: {e}")
                            finally:
                                scheduler.end_comm()
                            break

                        else:
                            continue  # 等待 send_last 被 set()

                    # ===========================================================
                    # ================ Collective  ========================
                    # ===========================================================
                    scheduler.begin_comm(overlap=False)
                    try:
                        print(f"[Device {device_id}] Performing collective {comm_type} with group={comm_group}")

                        # --- Simulated communication delay ---
                        with global_lock:
                            comm_time = link_manage.get_communication_time(task_node)
                            local_time[0] += comm_time
                            local_time_list[device_id] = local_time[0]
                        print(f"[Device {device_id}] Collective {comm_type} done. +{comm_time}, local_time={local_time[0]}")

                        # --- Barrier Synchronization ---
                        print(f"[Device {device_id}] Waiting at barrier for {group_key} (round={round_id})")
                        barrier.wait()
                        print(f"[Device {device_id}] Barrier released for {group_key} (round={round_id})")

                        # --- Synchronize local_time to group max ---
                        with global_lock:
                            group_times = []
                            for r in comm_group:
                                val = local_time_list[r]
                                if isinstance(val, list):
                                    val = val[0]
                                group_times.append(val)
                            # 判断所有设备时间是不是相同，不相同就得同步
                            group_max_time = max(group_times)
                            for r in comm_group:
                                if local_time_list[r] != group_max_time:
                                    local_time_list[r] = group_max_time
                                    
                            local_time[0] =  local_time_list[device_id]
                            print(f"[DEBUG][Device {device_id}] Synced local_time={local_time[0]} to max={group_max_time}")
                            # --- Register received tensor lifecycle ---
                            tensor_life = Memory_manage.LifeTime()
                            tensor_life.start_time = local_time[0]
                            mem_dict[task_node.name] = tensor_life
                            print(f"[DEBUG][Device {device_id}] Registered received tensor {task_node.name} at time {local_time[0]}")

                            # --- Check input tensors for release ---
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
                                        print(f"[DEBUG][Device {device_id}] Released memory after collective for {inp.name} at {local_time[0]}")
                        barrier.wait()
                        barrier_rounds[group_key] += 1
                    except threading.BrokenBarrierError:
                        print(f"[WARN][Device {device_id}] Barrier timeout for group {group_key}")
                    except Exception as e:
                        print(f"[ERROR][Device {device_id}] Collective/Recv failed: {e}")
                    finally:
                        scheduler.end_comm()
                    break
                else:
                    pass
                    # time.sleep(0.001)   

    print(f"[Device {device_id}] Communication thread finished.")
    return None




# ----------------------------
# --- Memory Thread ---
# ----------------------------
def memory_thread(device_id, mem_dict, stop_event: Event) -> None:
    print(f"[Device {device_id}] Memory thread started.")
    memory_manage = Memory_manage.Memory(mem_dict)
    while not stop_event.is_set():
        # print(f"[Device {device_id}] Managing memory...")
        pass
    print(f"[Device {device_id}] Generating memory report.")
    memory_manage.print_memory_report(device_id)
    return None



# ----------------------------
# --- Device Process ---
# ----------------------------
def device_process(rank, 
                   sched_queue, 
                   local_time_list, 
                   link_manager, 
                   group_barriers, 
                   pp_stage_gap, 
                   global_lock) -> double:
    
    print(f"[Device {rank}] Process started.")
    local_time = [0.0]
    compute_q = Queue()
    comm_q = Queue()
    mem_dict = {}
    stop_event = threading.Event()
    # recv_first = [False,]
    recv_first = threading.Event()
    send_last = threading.Event()


    scheduler = DeviceScheduler()
    lock = threading.Lock()   # 新增锁
    print(f"[DEBUG][Device {rank}] Initialized local queues. Total tasks={len(sched_queue)}")

    recv_placeholder_registered = False
    while sched_queue:
        task = sched_queue.popleft()
        if task.op == 'placeholder':
            if f'hccl_recv_' not in task.name:
                tensor_life = Memory_manage.LifeTime()
                with lock:
                    tensor_life.start_time = local_time[0]
                    mem_dict[task.name] = tensor_life
                print(f"[DEBUG][Device {rank}] Registered placeholder tensor {task.name}")
            else:
                tensor_life = Memory_manage.LifeTime()
                tensor_life.wait = True
                with lock:
                    mem_dict[task.name] = tensor_life
                if not recv_placeholder_registered:
                    comm_q.put('Recv_placeholders')
                    recv_placeholder_registered = True
                print(f"[DEBUG][Device {rank}] Registered recv placeholder {task.name}")
        
        elif task.op == 'call_function':
            task_target_name = str(task.target.__name__) if hasattr(task.target, '__name__') else str(task.target)
        
            if 'hccl' in task_target_name.lower():
                comm_q.put(task)
                print(f"[DEBUG][Device {rank}] Queued comm task {task.name}")
            else:
                compute_q.put(task)
                print(f"[DEBUG][Device {rank}] Queued compute task {task.name}")
            
        elif task.op == 'output':
            with lock:
                mem_dict[task.name] = task
            print(f"[DEBUG][Device {rank}] Output node {task.name} registered.")

    if not recv_placeholder_registered:
        # recv_first[0] = True
        recv_first.set()
    
    compute_thread_obj = threading.Thread(target=compute_thread, args=(rank, compute_q, local_time, mem_dict, local_time_list, scheduler, lock, global_lock, recv_first, send_last))
    comm_thread_obj = threading.Thread(target=comm_thread, args=(rank, comm_q, local_time, mem_dict, local_time_list, link_manager, group_barriers, scheduler, pp_stage_gap, lock, global_lock, recv_first, send_last))
    mem_thread_obj = threading.Thread(target=memory_thread, args=(rank, mem_dict, stop_event))

    group_barriers[f"All-devices-sync"].wait()
    print(f"[Device {rank}] Starting Communication threads...")
    comm_thread_obj.start()

    group_barriers[f"All-devices-sync"].wait()
    print(f"[Device {rank}] Starting Compute threads...")
    compute_thread_obj.start()

    group_barriers[f"All-devices-sync"].wait()
    print(f"[Device {rank}] Starting Memory threads...")
    mem_thread_obj.start()

    group_barriers[f"All-devices-sync"].wait()

    while True:
        if compute_q.empty() and comm_q.empty():
            print(f"[DEBUG][Device {rank}] All local queues empty.")
            break

        
    print(f"[Device {rank}] Waiting for threads to finish...")
    compute_thread_obj.join()
    comm_thread_obj.join()

    # 通知 memory_thread 停止
    stop_event.set()
    mem_thread_obj.join()
    print(f"[Device {rank}] Process finished. Local time={local_time[0]}")
    with global_lock:
        local_time_list[rank] = local_time[0]
    
    return local_time[0]
