import os
import pandas as pd
from pyvis.network import Network

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SHARE_DATA_DIR = os.path.join(ROOT_DIR, "share_data")


def get_node_style(score: float):
    """Quy ước màu sắc và trạng thái theo điểm thành thạo GAT"""
    if score < 0.50:
        return "#FF4B4B", "#C0392B", "Lỗ hổng kiến thức (Cần ôn lại ngay)"
    elif score < 0.70:
        return "#FFA500", "#D35400", "Mức trung bình (Cần củng cố thêm)"
    else:
        return "#2ECC71", "#27AE60", "Đã nắm vững"


def build_knowledge_graph_html(mastery_scores: dict[str, float], selected_chapter: str = "Tất cả") -> str:
    """Tạo mã HTML đồ thị tương tác PyVis cho 27 Concepts và các cạnh tiên quyết"""
    concepts_df = pd.read_csv(os.path.join(SHARE_DATA_DIR, "concepts.csv"))
    edges_df = pd.read_csv(os.path.join(SHARE_DATA_DIR, "edges.csv"))

    if selected_chapter != "Tất cả":
        concepts_df = concepts_df[concepts_df["chapter"] == selected_chapter]

    valid_cids = set(concepts_df["concept_id"].tolist())

    net = Network(
        height="580px",
        width="100%",
        directed=True,
        bgcolor="#FFFFFF",
        font_color="#1E1E1E",
    )

    # Cấu hình vật lý mượt mà, dễ nhìn theo luồng học tập
    net.set_options("""
    {
      "layout": {
        "hierarchical": {
          "enabled": false
        }
      },
      "physics": {
        "forceAtlas2Based": {
          "gravitationalConstant": -65,
          "centralGravity": 0.01,
          "springLength": 135,
          "springConstant": 0.08
        },
        "maxVelocity": 50,
        "solver": "forceAtlas2Based",
        "timestep": 0.35,
        "stabilization": { "iterations": 150 }
      },
      "interaction": {
        "hover": true,
        "tooltipDelay": 100
      }
    }
    """)

    # Thêm các nút Concept
    for _, row in concepts_df.iterrows():
        cid = row["concept_id"]
        cname = row["concept_name"]
        chapter = row["chapter"]
        score = float(mastery_scores.get(cid, 0.5))
        bg_color, border_color, status_text = get_node_style(score)

        tooltip = (
            f"<b>{cid}: {cname}</b><br>"
            f"<i>{chapter}</i><br>"
            f"Điểm thành thạo AI: <b>{score * 100:.1f}%</b><br>"
            f"Trạng thái: <b>{status_text}</b>"
        )

        net.add_node(
            cid,
            label=f"{cid}\n({score*100:.0f}%)",
            title=tooltip,
            color={
                "background": bg_color,
                "border": border_color,
                "highlight": {"background": bg_color, "border": "#000000"},
            },
            shape="dot",
            size=24 if score < 0.5 else 20,
            font={"size": 13, "face": "Arial", "bold": True},
        )

    # Thêm các cạnh quan hệ tiên quyết (Concept -> Concept)
    prereq_edges = edges_df[edges_df["relation"] == "prerequisite"]
    for _, edge in prereq_edges.iterrows():
        src, dst = edge["source_id"], edge["target_id"]
        if src in valid_cids and dst in valid_cids:
            net.add_edge(
                src,
                dst,
                arrows="to",
                color={"color": "#95A5A6", "highlight": "#2C3E50"},
                width=1.8,
                title=f"Kiến thức tiên quyết: {src} ➔ {dst}",
            )

    return net.generate_html()