import os
import sys
import pandas as pd
from pyvis.network import Network

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SHARE_DATA_DIR = os.path.join(ROOT_DIR, "share_data")
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import data_pipeline.database as db


def get_node_style(score: float):
    """Quy ước màu sắc và trạng thái theo điểm thành thạo GAT"""
    if score < 0.50:
        return "#FF4B4B", "#C0392B", "Lỗ hổng kiến thức (< 50%)"
    elif score < 0.70:
        return "#FFA500", "#D35400", "Cần củng cố (50% - 69%)"
    else:
        return "#2ECC71", "#27AE60", "Đã nắm vững (≥ 70%)"


def shorten_name(name: str, max_len: int = 24) -> str:
    clean = name.split("(")[0].strip()
    return clean if len(clean) <= max_len else clean[:max_len - 2].rstrip() + "..."


def generate_concept_explanations(
    mastery_scores: dict[str, float],
    grade_details: list[dict] | None = None,
    quiz_questions: list[dict] | None = None,
) -> dict[str, dict]:
    """
    Sinh lời giải thích Explainable AI trực quan, dễ hiểu theo quan hệ nhân quả:
    - Chỉ rõ Câu số mấy (kèm trích dẫn câu hỏi) đã làm sai / đúng / bỏ trống
    - Truy vết cả chiều xuôi (hổng nền tảng kéo theo nâng cao) lẫn chiều ngược (sai nâng cao suy ra hổng gốc rễ)
    - Cảnh báo các chủ đề phía sau sẽ bị ảnh hưởng nếu chưa khắc phục
    """
    all_concepts = db.get_all_concepts()
    c_map = {c["concept_id"]: c for c in all_concepts}

    edges_df = pd.read_csv(os.path.join(SHARE_DATA_DIR, "edges.csv"))
    questions_df = pd.read_csv(os.path.join(SHARE_DATA_DIR, "questions.csv"))
    q_text_map = dict(zip(questions_df["question_id"], questions_df["question_text"]))

    test_edges = edges_df[edges_df["relation"] == "tests"]
    prereq_edges = edges_df[edges_df["relation"] == "prerequisite"]

    q_order_map = {}
    if quiz_questions:
        for idx, q in enumerate(quiz_questions, start=1):
            q_order_map[q["question_id"]] = idx

    def format_q_ref(qid: str) -> str:
        prefix = f"Câu {q_order_map[qid]} ({qid})" if qid in q_order_map else qid
        raw_txt = str(q_text_map.get(qid, "")).strip()
        short_txt = raw_txt if len(raw_txt) <= 55 else raw_txt[:52].rstrip() + "..."
        return f"<b>{prefix}</b> <i>\"{short_txt}\"</i>" if short_txt else f"<b>{prefix}</b>"

    q_status = {}
    if grade_details:
        for item in grade_details:
            q_status[item["question_id"]] = "ĐÚNG" if item["is_correct"] else "SAI"

    c_to_qs = test_edges.groupby("target_id")["source_id"].apply(list).to_dict()
    c_to_prereqs = prereq_edges.groupby("target_id")["source_id"].apply(list).to_dict()
    c_to_nexts = prereq_edges.groupby("source_id")["target_id"].apply(list).to_dict()

    c_wrong_qs = {
        cid: [q for q in c_to_qs.get(cid, []) if q_status.get(q) == "SAI"]
        for cid in c_map
    }

    explanations = {}
    for cid, c_info in c_map.items():
        score = float(mastery_scores.get(cid, 0.5))
        linked_qs = c_to_qs.get(cid, [])
        prereqs = c_to_prereqs.get(cid, [])
        nexts = c_to_nexts.get(cid, [])

        wrong_qs = c_wrong_qs.get(cid, [])
        right_qs = [q for q in linked_qs if q_status.get(q) == "ĐÚNG"]
        tested_in_quiz = [q for q in linked_qs if q in q_order_map]
        skipped_in_quiz = [q for q in tested_in_quiz if q not in q_status]

        weak_prereqs = [p for p in prereqs if mastery_scores.get(p, 0.5) < 0.50]
        mid_prereqs = [p for p in prereqs if 0.50 <= mastery_scores.get(p, 0.5) < 0.70]
        weak_nexts = [n for n in nexts if mastery_scores.get(n, 0.5) < 0.50]

        reasons = []
        if score < 0.50:
            category = "RED"
            if wrong_qs:
                q_desc = "; ".join([format_q_ref(q) for q in wrong_qs])
                reasons.append(
                    f"<b>Nguyên nhân trực tiếp:</b> Bạn đã trả lời <b>SAI</b> {q_desc}, "
                    f"cho thấy chưa nắm chắc khái niệm cốt lõi của phần này."
                )

            if weak_prereqs:
                p_details = []
                for p in weak_prereqs:
                    p_name = shorten_name(c_map[p]["concept_name"], 28)
                    if c_wrong_qs.get(p):
                        p_q = ", ".join([
                            f"Câu {q_order_map[q]} ({q})" if q in q_order_map else q
                            for q in c_wrong_qs[p]
                        ])
                        p_details.append(f"<b>{p} ({p_name})</b> [do sai {p_q}]")
                    else:
                        p_details.append(f"<b>{p} ({p_name}: {mastery_scores.get(p, 0.5)*100:.0f}%)</b>")
                reasons.append(
                    f"<b>Ảnh hưởng dây chuyền (Quan hệ tiên quyết):</b> Chủ đề này đòi hỏi phải vững {', '.join(p_details)} trước. "
                    f"Vì phần nền tảng đó đang bị hổng nên mô hình GAT đánh giá chủ đề này cũng bị kéo xuống mức yếu."
                )

            if not wrong_qs and not weak_prereqs:
                if skipped_in_quiz:
                    sk_str = ", ".join([f"Câu {q_order_map[q]} ({q})" for q in skipped_in_quiz])
                    reasons.append(f"<b>Trạng thái bài làm:</b> Bạn đã <b>bỏ trống</b> {sk_str} thuộc chủ đề này.")
                if weak_nexts:
                    n_details = []
                    for n in weak_nexts:
                        n_name = shorten_name(c_map[n]["concept_name"], 28)
                        if c_wrong_qs.get(n):
                            n_q = ", ".join([
                                f"Câu {q_order_map[q]} ({q})" if q in q_order_map else q
                                for q in c_wrong_qs[n]
                            ])
                            n_details.append(f"<b>{n} ({n_name})</b> [sai {n_q}]")
                        else:
                            n_details.append(f"<b>{n} ({n_name})</b>")
                    reasons.append(
                        f"<b>Suy luận ngược từ kiến thức kế tiếp:</b> Dù bạn chưa làm câu hỏi trực tiếp ở đây, "
                        f"nhưng vì bạn làm sai phần ứng dụng phía sau là {', '.join(n_details)}, "
                        f"mô hình GAT suy luận rằng lỗ hổng bắt nguồn từ chính kiến thức gốc này."
                    )
                else:
                    reasons.append(
                        "<b>Lan truyền đồ thị GAT:</b> Bạn chưa trả lời câu hỏi trực tiếp và mạng nơ-ron đồ thị "
                        "phát hiện sự sụt giảm điểm ở cụm kiến thức liên quan xung quanh."
                    )

            if nexts:
                next_names = ", ".join([
                    f"<b>{n}</b> ({shorten_name(c_map[n]['concept_name'], 22)})"
                    for n in nexts[:3]
                ])
                reasons.append(
                    f"<b>Cảnh báo ảnh hưởng tiếp theo:</b> Nếu không ôn lại phần này ngay, "
                    f"bạn sẽ gặp khó khăn khi học sang: {next_names}."
                )

        elif score < 0.70:
            category = "YELLOW"
            if right_qs:
                q_desc = "; ".join([format_q_ref(q) for q in right_qs])
                if weak_prereqs or mid_prereqs:
                    p_list = ", ".join([
                        f"<b>{p}</b> ({shorten_name(c_map[p]['concept_name'], 24)}: {mastery_scores.get(p, 0.5)*100:.0f}%)"
                        for p in (weak_prereqs + mid_prereqs)
                    ])
                    reasons.append(
                        f"<b>Làm đúng nhưng nền tảng chưa chắc:</b> Bạn đã trả lời <b>ĐÚNG</b> {q_desc}, "
                        f"tuy nhiên các kiến thức tiên quyết đứng trước là {p_list} đang bị hổng hoặc mới ở mức trung bình. "
                        f"Vì vậy GAT xếp vào nhóm <b>Cần củng cố</b> để tránh học vẹt mà thiếu gốc."
                    )
                else:
                    reasons.append(
                        f"<b>Cần luyện tập thêm để đạt mức Vững (≥ 70%):</b> Bạn đã trả lời <b>ĐÚNG</b> {q_desc}, "
                        f"nhưng do bài test nhanh chỉ kiểm tra 1 câu đại diện nên mô hình GAT giữ điểm ở mức an toàn ({score*100:.0f}%). "
                        f"Hãy làm thêm bài tập chuyên sâu để xác nhận nắm vững hoàn toàn."
                    )
            else:
                if skipped_in_quiz:
                    sk_str = ", ".join([f"Câu {q_order_map[q]} ({q})" for q in skipped_in_quiz])
                    reasons.append(
                        f"<b>Chưa đủ dữ liệu kiểm chứng:</b> Bạn đã <b>bỏ trống</b> {sk_str} trong bài làm. "
                        f"Mô hình GAT tạm tính mức thành thạo trung tính ({score*100:.0f}%) dựa trên các chủ đề lân cận — "
                        f"bạn nên làm thử câu hỏi phần này để hệ thống chẩn đoán chính xác."
                    )
                else:
                    reasons.append("Chưa có câu hỏi kiểm tra trực tiếp trong đề; điểm số được ước lượng trung tính từ cấu trúc đồ thị.")

        else:
            category = "GREEN"
            if right_qs:
                q_desc = "; ".join([format_q_ref(q) for q in right_qs])
                reasons.append(
                    f"<b>Nắm vững kiến thức:</b> Bạn đã trả lời <b>ĐÚNG</b> {q_desc} "
                    f"đồng thời các kiến thức nền tảng liên quan đều đạt mức tốt."
                )
            else:
                reasons.append(
                    "<b>Suy luận tích cực từ đồ thị:</b> Các chủ đề nền tảng và mở rộng liên kết với phần này đều đạt kết quả cao."
                )

        explanations[cid] = {
            "concept_id": cid,
            "concept_name": c_info["concept_name"],
            "chapter": c_info["chapter"],
            "score": score,
            "category": category,
            "reasons": reasons,
        }

    return explanations


def build_knowledge_graph_html(
    mastery_scores: dict[str, float],
    selected_chapter: str = "Tất cả",
    grade_details: list[dict] | None = None,
) -> str:
    """Tạo mã HTML đồ thị tương tác PyVis kèm thống kê câu làm đúng/sai/chưa làm trong tooltip"""
    all_concepts = db.get_all_concepts()
    if selected_chapter != "Tất cả":
        all_concepts = [c for c in all_concepts if c["chapter"] == selected_chapter]

    valid_cids = {c["concept_id"] for c in all_concepts}

    edges_df = pd.read_csv(os.path.join(SHARE_DATA_DIR, "edges.csv"))
    prereq_edges = edges_df[edges_df["relation"] == "prerequisite"]
    test_edges = edges_df[edges_df["relation"] == "tests"]

    prereq_in = prereq_edges.groupby("target_id")["source_id"].apply(list).to_dict()
    prereq_out = prereq_edges.groupby("source_id")["target_id"].apply(list).to_dict()
    c_to_qs = test_edges.groupby("target_id")["source_id"].apply(list).to_dict()

    q_status_map = {
        d["question_id"]: ("Đúng" if d.get("is_correct") else "Sai")
        for d in (grade_details or [])
    }

    net = Network(
        height="620px",
        width="100%",
        directed=True,
        bgcolor="#FAFAFA",
        font_color="#1E1E1E",
    )

    net.set_options("""
    {
      "layout": {
        "hierarchical": {
          "enabled": false
        }
      },
      "physics": {
        "forceAtlas2Based": {
          "gravitationalConstant": -85,
          "centralGravity": 0.01,
          "springLength": 155,
          "springConstant": 0.08
        },
        "maxVelocity": 50,
        "solver": "forceAtlas2Based",
        "timestep": 0.35,
        "stabilization": { "iterations": 180 }
      },
      "interaction": {
        "hover": true,
        "tooltipDelay": 60,
        "navigationButtons": true
      }
    }
    """)

    for c in all_concepts:
        cid = c["concept_id"]
        cname = c["concept_name"]
        chapter = c["chapter"]
        score = float(mastery_scores.get(cid, 0.5))
        bg_color, border_color, status_text = get_node_style(score)

        in_list = prereq_in.get(cid, [])
        out_list = prereq_out.get(cid, [])
        in_str = ", ".join(in_list) if in_list else "Không có (Kiến thức gốc)"
        out_str = ", ".join(out_list) if out_list else "Không có (Kiến thức đích)"

        linked_qs = c_to_qs.get(cid, [])
        done_right = [q for q in linked_qs if q_status_map.get(q) == "Đúng"]
        done_wrong = [q for q in linked_qs if q_status_map.get(q) == "Sai"]
        not_done = [q for q in linked_qs if q not in q_status_map]

        q_summary = (
            f"Câu hỏi liên quan: Đúng {len(done_right)} ({', '.join(done_right) if done_right else '0'}) | "
            f"Sai {len(done_wrong)} ({', '.join(done_wrong) if done_wrong else '0'}) | "
            f"Chưa làm {len(not_done)}"
        )

        tooltip = (
            f"{cid}: {cname}\n"
            f"Thuộc: {chapter}\n"
            f"Thành thạo: {score * 100:.1f}% ({status_text})\n"
            f"{q_summary}\n"
            f"Cần học trước: {in_str}\n"
            f"Mở khóa tiếp theo: {out_str}"
        )

        short_label = f"{cid}: {shorten_name(cname)}\n({score * 100:.0f}%)"

        net.add_node(
            cid,
            label=short_label,
            title=tooltip,
            color={
                "background": bg_color,
                "border": border_color,
                "highlight": {"background": bg_color, "border": "#000000"},
            },
            shape="dot",
            size=26 if score < 0.5 else 21,
            font={"size": 12, "face": "Arial", "bold": True},
        )

    for _, edge in prereq_edges.iterrows():
        src, dst = edge["source_id"], edge["target_id"]
        if src in valid_cids and dst in valid_cids:
            net.add_edge(
                src,
                dst,
                arrows="to",
                color={"color": "#95A5A6", "highlight": "#2C3E50"},
                width=2.0,
                title=f"Tiên quyết: {src} ➔ {dst}",
            )

    return net.generate_html()