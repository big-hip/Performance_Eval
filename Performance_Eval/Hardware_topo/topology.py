import json
import networkx as nx

class TopologyGraph:
    def __init__(self):
        # 初始化空无向图
        self.graph = nx.Graph()

    def load_from_json(self, json_path: str):
        """
        从JSON文件加载拓扑配置，并构建网络图。
        JSON中支持基础节点和边，以及六种基础网络结构：
        ring, star, tree, bus, mesh, hybrid。
        """
        with open(json_path, 'r') as f:
            data = json.load(f)
           # === 第一步：统一添加所有节点（包括六种拓扑中定义的） ===

        # 添加基础 nodes
        for node in data.get("nodes", []):
            self._add_unique_node(node["id"], node["device_type"], node["host_node"])

        # 添加 star 中的节点
        for star in data.get("star", []):
            self._add_unique_node(star["center_id"], "unknown", star["center_host"])
            for leaf in star["leaf_ids"]:
                self._add_unique_node(leaf["id"], "unknown", leaf["host_node"])

        # 添加 ring 中 device_nums 生成节点和直接定义的节点
        for ring_group in data.get("ring", []):
            for node in ring_group:
                if "device_nums" in node:
                    for i in range(node["device_nums"]):
                        generated_id = f"{node['device_type']}{i+1}"
                        self._add_unique_node(generated_id, node["device_type"], node["host_node"])
                else:
                    self._add_unique_node(node["id"], node["device_type"], node["host_node"])

        # 添加 tree 中所有节点
        for tree in data.get("tree", []):
            for node in tree.get("nodes", []):
                self._add_unique_node(node["id"], node["device_type"], node["host_node"])

        # 添加 bus 中所有节点
        for bus in data.get("bus", []):
            for node in bus.get("nodes", []):
                self._add_unique_node(node["id"], node["device_type"], node["host_node"])

        # 添加 mesh 中所有节点
        for mesh in data.get("mesh", []):
            for node in mesh.get("nodes", []):
                self._add_unique_node(node["id"], node["device_type"], node["host_node"])

        # 添加 hybrid 中的所有节点（递归处理每类结构）
        for hybrid in data.get("hybrid", []):
            for ring_group in hybrid.get("ring", []):
                for node in ring_group:
                    if "device_nums" in node:
                        for i in range(node["device_nums"]):
                            generated_id = f"{node['device_type']}{i+1}"
                            self._add_unique_node(generated_id, node["device_type"], node["host_node"])
                    else:
                        self._add_unique_node(node["id"], node["device_type"], node["host_node"])
            for star in hybrid.get("star", []):
                self._add_unique_node(star["center_id"], "unknown", star["center_host"])
                for leaf in star["leaf_ids"]:
                    self._add_unique_node(leaf["id"], "unknown", leaf["host_node"])
            for tree in hybrid.get("tree", []):
                for node in tree.get("nodes", []):
                    self._add_unique_node(node["id"], node["device_type"], node["host_node"])
            for bus in hybrid.get("bus", []):
                for node in bus.get("nodes", []):
                    self._add_unique_node(node["id"], node["device_type"], node["host_node"])
            for mesh in hybrid.get("mesh", []):
                for node in mesh.get("nodes", []):
                    self._add_unique_node(node["id"], node["device_type"], node["host_node"])


        # 加载基础边（现在要求 source/target 为 dict）
        for edge in data.get("edges", []):
            self._add_unique_edge(
                edge['source']['id'], edge['source']['host_node'],
                edge['target']['id'], edge['target']['host_node'],
                edge["link_type"], edge["bandwidth"]
            )

        # 加载环形拓扑，ring是二维数组，每个元素是节点列表
        for ring_group in data.get("ring", []):
            if isinstance(ring_group, list):
                self._build_ring_topology(ring_group)
            else:
                raise ValueError(f"Expected ring group to be list but got {type(ring_group)}")

        # 加载星形拓扑，star是列表，元素是字典
        for star_group in data.get("star", []):
            if isinstance(star_group, dict):
                self._build_star_topology(star_group)
            else:
                raise ValueError(f"Expected star group to be dict but got {type(star_group)}")

        # 加载树形拓扑
        for tree_group in data.get("tree", []):
            if isinstance(tree_group, dict):
                self._build_tree_topology(tree_group)
            else:
                raise ValueError(f"Expected tree group to be dict but got {type(tree_group)}")

        # 加载总线拓扑
        for bus_group in data.get("bus", []):
            if isinstance(bus_group, dict):
                self._build_bus_topology(bus_group)
            else:
                raise ValueError(f"Expected bus group to be dict but got {type(bus_group)}")

        # 加载网状拓扑
        for mesh_group in data.get("mesh", []):
            if isinstance(mesh_group, dict):
                self._build_mesh_topology(mesh_group)
            else:
                raise ValueError(f"Expected mesh group to be dict but got {type(mesh_group)}")

        # 加载混合拓扑
        for hybrid_group in data.get("hybrid", []):
            if isinstance(hybrid_group, dict):
                self._build_hybrid_topology(hybrid_group)
            else:
                raise ValueError(f"Expected hybrid group to be dict but got {type(hybrid_group)}")

    def export_to_json(self, output_path: str):
        """
        导出当前图结构为JSON文件，仅包含nodes和edges信息，方便保存和共享。
        """
        nodes = []
        for node, attr in self.graph.nodes(data=True):
            nodes.append({
                "id": attr["label_name"],
                "device_type": attr["device_type"],
                "host_node": attr["host_node"]
            })

        edges = []
        for src, tgt, attr in self.graph.edges(data=True):
            src_id = self.graph.nodes[src]["label_name"]
            src_host = self.graph.nodes[src]["host_node"]
            tgt_id = self.graph.nodes[tgt]["label_name"]
            tgt_host = self.graph.nodes[tgt]["host_node"]
            # 过滤自环
            if src_id == tgt_id and src_host == tgt_host:
                continue
            edges.append({
                "source": src_id,
                "target": tgt_id,
                "link_type": attr["link_type"],
                "bandwidth": attr["bandwidth"]
            })

        with open(output_path, 'w') as f:
            json.dump({"nodes": nodes, "edges": edges}, f, indent=2)

    def _get_host_node(self, node_id, data):
        """
        根据节点id从数据中查找其host_node，用于构建唯一节点标识。
        """
        for node in data.get("nodes", []):
            if node["id"] == node_id:
                return node["host_node"]
        return ""

    def _add_unique_node(self, node_id, device_type, host_node):
        """
        添加唯一节点到图，唯一ID为 f"{node_id}_{host_node}"，避免重复。
        """
        unique_id = f"{node_id}_{host_node}"
        if unique_id not in self.graph:
            self.graph.add_node(
                unique_id,
                label_name=node_id,
                device_type=device_type,
                host_node=host_node
            )

    def _add_unique_edge(self, src_id, src_host, tgt_id, tgt_host, link_type, bandwidth):
        """
        添加唯一边到图，避免重复边。边属性包含带宽和连接类型。
        """
        src = f"{src_id}_{src_host}"
        tgt = f"{tgt_id}_{tgt_host}"
        if not self.graph.has_edge(src, tgt):
            self.graph.add_edge(
                src,
                tgt,
                link_type=link_type,
                bandwidth=bandwidth
            )

    # --- 各种拓扑构建函数 ---

    def _build_ring_topology(self, ring_nodes):
        """
        构建环形拓扑。ring_nodes是列表，支持device_nums批量生成节点。
        """
        if not ring_nodes:
            return
        # 展开device_nums节点
        nodes_expanded = []
        for node in ring_nodes:
            if isinstance(node, dict) and "device_nums" in node:
                # 批量生成
                nodes_expanded.extend([
                    {
                        "id": f"{node['device_type']}{i+1}",
                        "device_type": node["device_type"],
                        "host_node": node["host_node"],
                        "link_type": node["link_type"],
                        "bandwidth": node["bandwidth"]
                    }
                    for i in range(node["device_nums"])
                ])
            else:
                nodes_expanded.append(node)

        # 添加节点
        for node in nodes_expanded:
            self._add_unique_node(node["id"], node["device_type"], node["host_node"])

        # 添加环形边，支持每条边独立带宽和连接方式
        n = len(nodes_expanded)
        for i in range(n):
            src = nodes_expanded[i]
            tgt = nodes_expanded[(i + 1) % n]
            # 优先使用src的link_type/bandwidth字段
            link_type = src.get("link_type") or tgt.get("link_type")
            bandwidth = src.get("bandwidth") or tgt.get("bandwidth")
            self._add_unique_edge(
                src["id"], src["host_node"],
                tgt["id"], tgt["host_node"],
                link_type,
                bandwidth
            )

    def _build_star_topology(self, star):
        """
        构建星形拓扑。star中叶子节点支持带link_type和bandwidth字段。
        """
        center = star['center_id']
        center_host = star['center_host']
        # 中心节点类型未知时用unknown
        self._add_unique_node(center, 'unknown', center_host)

        for leaf in star['leaf_ids']:
            if isinstance(leaf, dict):
                leaf_id = leaf["id"]
                leaf_host = leaf["host_node"]
                # 添加叶节点
                self._add_unique_node(leaf_id, 'unknown', leaf_host)
                # 叶子节点可以有独立连接类型和带宽，否则用star的默认
                link_type = leaf.get("link_type", star.get("link_type"))
                bandwidth = leaf.get("bandwidth", star.get("bandwidth"))
                self._add_unique_edge(center, center_host, leaf_id, leaf_host, link_type, bandwidth)
            else:
                raise ValueError("Each leaf in star must be dict with 'id' and 'host_node'")

    def _build_tree_topology(self, tree):
        """
        构建树形拓扑。树节点显式定义，边支持独立带宽和连接方式。
        """
        # 添加所有节点
        for node in tree.get("nodes", []):
            self._add_unique_node(node["id"], node["device_type"], node["host_node"])
        # 添加所有边，边带宽和连接方式可能在边定义中
        for edge in tree.get("edges", []):
            src = edge["source"]
            tgt = edge["target"]
            link_type = edge.get("link_type", tree.get("link_type"))
            bandwidth = edge.get("bandwidth", tree.get("bandwidth"))
            self._add_unique_edge(
                src["id"], src["host_node"],
                tgt["id"], tgt["host_node"],
                link_type, bandwidth
            )

    def _build_bus_topology(self, bus):
        """
        构建总线拓扑。
        支持显式指定边集合（含带宽和连接方式），也支持默认线性连接带宽和连接方式。
        """
        nodes = bus.get("nodes", [])
        # 添加节点
        for node in nodes:
            self._add_unique_node(node["id"], node["device_type"], node["host_node"])

        # 如果显式定义边，则用边定义，否则默认线性连接
        edges = bus.get("edges")
        if edges:
            for edge in edges:
                src = edge["source"]
                tgt = edge["target"]
                link_type = edge["link_type"]
                bandwidth = edge["bandwidth"]
                self._add_unique_edge(
                    src["id"], src["host_node"],
                    tgt["id"], tgt["host_node"],
                    link_type, bandwidth
                )
        else:
            # 默认线性连接邻接节点
            n = len(nodes)
            for i in range(n - 1):
                self._add_unique_edge(
                    nodes[i]["id"], nodes[i]["host_node"],
                    nodes[i + 1]["id"], nodes[i + 1]["host_node"],
                    bus.get("link_type"), bus.get("bandwidth")
                )

    def _build_mesh_topology(self, mesh):
        """
        mesh拓扑构建:
        - 所有节点两两全连接。
        - 如果mesh配置中有edges字段，且某条边被显式定义，则该边使用定义的link_type和bandwidth。
        - 否则该边使用mesh配置的默认link_type和bandwidth。
        """
        nodes = mesh["nodes"]
        for node in nodes:
            self._add_unique_node(node["id"], node["device_type"], node["host_node"])

        # 构建一个快速查找显式边配置的字典，key是(frozenset({src_id, tgt_id}))，value是边属性
        explicit_edges = {}
        if "edges" in mesh:
            for edge in mesh["edges"]:
                # 使用frozenset表示无向边（节点对）
                edge_key = frozenset({edge["source"]["id"], edge["target"]["id"]})
                explicit_edges[edge_key] = {
                    "link_type": edge.get("link_type", mesh.get("link_type")),
                    "bandwidth": edge.get("bandwidth", mesh.get("bandwidth"))
                }

        n = len(nodes)
        for i in range(n):
            for j in range(i + 1, n):
                src = nodes[i]
                tgt = nodes[j]
                edge_key = frozenset({src["id"], tgt["id"]})
                if edge_key in explicit_edges:
                    link_type = explicit_edges[edge_key]["link_type"]
                    bandwidth = explicit_edges[edge_key]["bandwidth"]
                else:
                    link_type = mesh.get("link_type")
                    bandwidth = mesh.get("bandwidth")

                self._add_unique_edge(
                    src["id"], src["host_node"],
                    tgt["id"], tgt["host_node"],
                    link_type,
                    bandwidth
                )

    def _build_hybrid_topology(self, hybrid):
        """
        构建混合拓扑，递归调用对应类型的构建函数。
        """
        if "ring" in hybrid:
            for ring_group in hybrid["ring"]:
                if isinstance(ring_group, list):
                    self._build_ring_topology(ring_group)
                else:
                    raise ValueError(f"Expected ring group to be list but got {type(ring_group)}")

        if "star" in hybrid:
            for star_group in hybrid["star"]:
                if isinstance(star_group, dict):
                    self._build_star_topology(star_group)
                else:
                    raise ValueError(f"Expected star group to be dict but got {type(star_group)}")

        if "tree" in hybrid:
            for tree_group in hybrid["tree"]:
                if isinstance(tree_group, dict):
                    self._build_tree_topology(tree_group)
                else:
                    raise ValueError(f"Expected tree group to be dict but got {type(tree_group)}")

        if "bus" in hybrid:
            for bus_group in hybrid["bus"]:
                if isinstance(bus_group, dict):
                    self._build_bus_topology(bus_group)
                else:
                    raise ValueError(f"Expected bus group to be dict but got {type(bus_group)}")

        if "mesh" in hybrid:
            for mesh_group in hybrid["mesh"]:
                if isinstance(mesh_group, dict):
                    self._build_mesh_topology(mesh_group)
                else:
                    raise ValueError(f"Expected mesh group to be dict but got {type(mesh_group)}")

    # --- 基础增删查改接口 ---

    def get_graph(self):
        """返回当前的NetworkX图对象"""
        return self.graph

    def add_node(self, node_id, device_type, host_node):
        """添加节点"""
        self._add_unique_node(node_id, device_type, host_node)

    def remove_node(self, node_id, host_node):
        """删除节点"""
        unique_id = f"{node_id}_{host_node}"
        if unique_id in self.graph:
            self.graph.remove_node(unique_id)

    def update_node(self, node_id, host_node, **kwargs):
        """更新节点属性"""
        unique_id = f"{node_id}_{host_node}"
        if unique_id in self.graph.nodes:
            self.graph.nodes[unique_id].update(kwargs)

    def get_node(self, node_id, host_node):
        """获取节点属性"""
        unique_id = f"{node_id}_{host_node}"
        return self.graph.nodes.get(unique_id)

    def add_edge(self, src_id, src_host, tgt_id, tgt_host, link_type, bandwidth):
        """添加边"""
        self._add_unique_edge(src_id, src_host, tgt_id, tgt_host, link_type, bandwidth)

    def remove_edge(self, src_id, src_host, tgt_id, tgt_host):
        """删除边"""
        src = f"{src_id}_{src_host}"
        tgt = f"{tgt_id}_{tgt_host}"
        if self.graph.has_edge(src, tgt):
            self.graph.remove_edge(src, tgt)

    def update_edge(self, src_id, src_host, tgt_id, tgt_host, **kwargs):
        """更新边属性"""
        src = f"{src_id}_{src_host}"
        tgt = f"{tgt_id}_{tgt_host}"
        if self.graph.has_edge(src, tgt):
            self.graph.edges[src, tgt].update(kwargs)

    def get_edge(self, src_id, src_host, tgt_id, tgt_host):
        """获取边属性"""
        src = f"{src_id}_{src_host}"
        tgt = f"{tgt_id}_{tgt_host}"
        return self.graph.edges.get((src, tgt))

    def list_nodes(self):
        """列出所有节点及属性"""
        return list(self.graph.nodes(data=True))

    def list_edges(self):
        """列出所有边及属性"""
        return list(self.graph.edges(data=True))



"""
    基础部分：
    - nodes: 节点列表，格式示例：
        [
            {"id": "CPU1", "device_type": "CPU", "host_node": "HostA"},
            ...
        ]
        说明：每个节点有唯一id，设备类型，所属主机名。

    - edges: 基础边列表，格式示例：
        [
             {
                "source": {"id": "CPU1", "host_node": "RingHost1"},
                "target": {"id": "CPU1", "host_node": "RingHost2"},
                "link_type": "PCIe",
                "bandwidth": 15.0
            }
            ...
        ]
        说明：source和target为节点id，带宽单位自定义（如Gbps），连接类型自由定义。

    六种基础拓扑结构：

    1. ring（环形拓扑）
        - 格式示例：
            "ring": [
            [
                {
                    "host_node": "HostR",
                    "device_nums": 4,
                    "device_type": "NPU",
                    "link_type": "HCCS",
                    "bandwidth": 20.0
                },
                {
                    "id": "RingSwitch1",
                    "device_type": "Switch",
                    "host_node": "HostR",
                    "link_type": "HCCS",
                    "bandwidth": 30.0
                }
            ],
            ...
            ]
        - 说明：
            每个元素是一个列表，表示一圈的节点。
            节点可以通过device_nums批量生成（id格式为 device_type+序号）。
            边连接相邻节点，支持每条边指定link_type和bandwidth。

    2. star（星型拓扑）
        - 格式示例：
            "star": [
            {
                "center_id": "STAR_SW1",
                "center_host": "StarCenter1",
                "leaf_ids": [
                    {"id": "STAR_NPU1", "host_node": "Node_A", "link_type": "ethernet", "bandwidth": 2.0},
                    {"id": "STAR_NPU2", "host_node": "Node_B"},
                    ...
                ],
                "link_type": "ethernet",
                "bandwidth": 2.0
            },
            ...
            ]
        - 说明：
            中心节点center_id和host必填。
            叶子节点leaf_ids是列表，支持叶节点单独指定连接类型和带宽，默认使用star的link_type和bandwidth。
            边为中心节点与每个叶子节点相连。

    3. tree（树形拓扑）
        - 格式示例：
            "tree": [
            {
                "nodes": [
                    {"id": "ROOT", "device_type": "Switch", "host_node": "Host1"},
                    {"id": "Child1", "device_type": "CPU", "host_node": "Host2"},
                    ...
                ],
                "edges": [
                    {"source": {"id": "ROOT", "host_node": "Host1"}, "target": {"id": "Child1", "host_node": "Host2"}, "link_type": "PCIe", "bandwidth": 10.0},
                    ...
                ],
                "link_type": "PCIe",
                "bandwidth": 10.0
            }
            ]
        - 说明：
            明确列出节点和边，边支持独立link_type和bandwidth，否则默认使用tree节点的配置。

    4. bus（总线拓扑）
        - 格式示例：
            "bus": [
            {
                "nodes": [
                    {"id": "BUS1", "device_type": "CPU", "host_node": "HostB"},
                    {"id": "BUS2", "device_type": "CPU", "host_node": "HostB"},
                    ...
                ],
                "edges": [    # 可选
                    {"source": {"id": "BUS1", "host_node": "HostB"}, "target": {"id": "BUS2", "host_node": "HostB"}, "link_type": "USB", "bandwidth": 5.0}
                ],
                "link_type": "USB",
                "bandwidth": 4.0
            }
            ]
        - 说明：
            如果edges定义了边，则使用edges中的边和配置；
            否则默认所有节点线性相连，边带宽和连接类型使用bus的link_type和bandwidth。

    5. mesh（网状拓扑）
        - 格式示例：
            "mesh": [
            {
                "nodes": [...],
                "edges": [   # 可选
                    {"source": {...}, "target": {...}, "link_type": "...", "bandwidth": ...}
                ],
                "link_type": "HCCS",
                "bandwidth": 50.0
            }
            ]
        - 说明：
            如果edges存在，则使用显式边配置；
            否则所有节点全连接，边带宽和连接类型使用mesh的link_type和bandwidth。

    6. hybrid（混合拓扑）
        - 格式示例：
            "hybrid": [
            {
                "ring": [...],
                "star": [...],
                "tree": [...],
                "bus": [...],
                "mesh": [...]
            }
            ]
        - 说明：
            混合拓扑支持任意组合多种基础拓扑，格式同对应类型。

    ---
    以上各拓扑均支持边的独立link_type和bandwidth，优先使用边定义的字段，否则用结构默认字段。
"""