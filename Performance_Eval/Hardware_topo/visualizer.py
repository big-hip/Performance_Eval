import networkx as nx
from networkx.drawing.nx_pydot import to_pydot
import hashlib

def color_from_string(s: str) -> str:
    h = hashlib.md5(s.encode()).hexdigest()
    r = int(h[0:2], 16)
    g = int(h[2:4], 16)
    b = int(h[4:6], 16)
    return f"#{r:02x}{g:02x}{b:02x}"

def export_to_dot(graph: nx.Graph, output_path: str) -> None:
    dot_graph = to_pydot(graph)

    for node in dot_graph.get_nodes():
        node_id = node.get_name().strip('"')
        attrs = graph.nodes[node_id]

        display_name = attrs.get("label_name", node_id)

        label_rows = [f'<TR><TD><B>{display_name}</B></TD></TR>', '<HR/>']
        for k, v in attrs.items():
            label_rows.append(f'<TR><TD ALIGN="LEFT">{k}: {v}</TD></TR>')

        html_label = (
            '<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0" CELLPADDING="4">'
            + "".join(label_rows) +
            '</TABLE>>'
        )
        node.set("label", html_label)

        device_type = attrs.get("device_type", "").upper()
        if device_type == "NPU":
            node.set_shape("box")
            node.set_style("rounded,filled")
        elif device_type == "CPU":
            node.set_shape("ellipse")
            node.set_style("filled")
        else:
            node.set_shape("box")
            node.set_style("filled")

        # 固定尺寸
        # node.set("width", "1.4")
        # node.set("height", "0.9")
        # node.set("fixedsize", "true")

        host_node = attrs.get("host_node", "")
        node.set_fillcolor(color_from_string(host_node))

    for edge in dot_graph.get_edges():
        src = edge.get_source()
        tgt = edge.get_destination()
        attrs = graph.get_edge_data(src, tgt)

        if attrs:
            edge.set_label("")
            edge.set("xlabel", " ".join(str(v) for v in attrs.values()))
        else:
            edge.set_label("")

    with open(output_path, "w") as f:
        f.write(dot_graph.to_string())
