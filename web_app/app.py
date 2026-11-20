import base64
import os
import random
import sqlite3
import sys
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import data_pipeline.database as db
from web_app.mock_ai import run_gnn_diagnosis, build_demo_user_answers
from web_app.graph_viz import build_knowledge_graph_html, generate_concept_explanations

st.set_page_config(
    page_title="Hệ thống Chẩn đoán Lỗ hổng Kiến thức QLDA CNTT",
    layout="wide",
)

# Khởi tạo Session State
default_states = {
    "quiz_questions": [],
    "quiz_option_orders": {},
    "session_id": None,
    "grade_result": None,
    "answered_count": 0,
    "mastery_scores": None,
    "loaded_history_details": None,
    "loaded_history_total_q": 0,
    "preview_pdf_path": None,
    "scroll_to_top_after_grade": False,
    "limit_per_concept": 1,
}
for k, v in default_states.items():
    if k not in st.session_state:
        st.session_state[k] = v


def escape_markdown(text: str) -> str:
    """Thoát ký tự $ để Streamlit không nhận nhầm thành công thức toán học LaTeX"""
    if not text:
        return ""
    return str(text).replace("$", r"\$")


def randomize_option_order():
    order = random.sample(["A", "B", "C", "D"], k=4)
    if order == ["A", "B", "C", "D"]:
        order[0], order[1] = order[1], order[0]
    return order


def load_new_quiz(limit_per_concept=1):
    for question in st.session_state.quiz_questions:
        st.session_state.pop(f"radio_{question['question_id']}", None)

    st.session_state.limit_per_concept = limit_per_concept
    st.session_state.quiz_questions = db.get_diagnostic_test(
        limit_per_concept=limit_per_concept, include_answer=False
    )
    st.session_state.quiz_option_orders = {
        question["question_id"]: randomize_option_order()
        for question in st.session_state.quiz_questions
    }
    st.session_state.session_id = db.start_test_session(user_id=None)
    st.session_state.grade_result = None
    st.session_state.answered_count = 0
    st.session_state.mastery_scores = None
    st.session_state.loaded_history_details = None
    st.session_state.loaded_history_total_q = 0
    st.session_state.scroll_to_top_after_grade = False


def get_all_test_history():
    """Truy vấn toàn bộ lịch sử làm bài kèm chi tiết từng câu trả lời và điểm GNN"""
    db_path = os.path.join(ROOT_DIR, "data_pipeline", "database.db")
    if not os.path.exists(db_path):
        return []
    with sqlite3.connect(db_path) as con:
        con.row_factory = sqlite3.Row
        tables = {
            r["name"]
            for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        }
        if "test_sessions" not in tables:
            return []

        rows = con.execute(
            """
            SELECT * FROM test_sessions
            WHERE submitted_at IS NOT NULL
            ORDER BY submitted_at DESC
            """
        ).fetchall()

        results = []
        for r in rows:
            d = dict(r)
            sid = d.get("session_id") if d.get("session_id") is not None else d.get("id")

            details_list = []
            for ans_tbl in ["test_answers", "session_answers", "user_answers", "submission_details", "answers"]:
                if ans_tbl in tables:
                    try:
                        a_rows = con.execute(
                            f"""
                            SELECT a.*, q.correct_answer
                            FROM {ans_tbl} a
                            LEFT JOIN questions q ON a.question_id = q.question_id
                            WHERE a.session_id = ?
                            """,
                            (sid,),
                        ).fetchall()
                        if a_rows:
                            for ar in a_rows:
                                ad = dict(ar)
                                details_list.append({
                                    "question_id": ad["question_id"],
                                    "chosen": ad.get("chosen_answer") or ad.get("selected_answer") or ad.get("user_answer") or ad.get("answer"),
                                    "correct_answer": ad.get("correct_answer"),
                                    "is_correct": bool(ad.get("is_correct", 0)),
                                })
                            break
                    except Exception:
                        pass

            correct_c = (
                sum(1 for item in details_list if item["is_correct"])
                if details_list
                else (
                    d.get("correct_count")
                    if d.get("correct_count") is not None
                    else d.get("num_correct")
                    if d.get("num_correct") is not None
                    else d.get("correct_answers", 0)
                )
            )
            answered_c = (
                len(details_list)
                if details_list
                else (
                    d.get("answered_count")
                    if d.get("answered_count") is not None
                    else d.get("total_answered")
                    if d.get("total_answered") is not None
                    else d.get("question_count")
                )
            )
            total_q_saved = d.get("total_questions") or (54 if (answered_c or 0) > 27 else 27)
            score_val = (
                round((correct_c / answered_c) * 10, 2)
                if answered_c and answered_c > 0
                else (
                    d.get("score_10")
                    if d.get("score_10") is not None
                    else d.get("score", 0.0)
                )
            )

            saved_scores = {}
            for tbl in tables:
                if "mastery" in tbl or "gnn" in tbl or "diagnosis" in tbl:
                    try:
                        cols = [c["name"] for c in con.execute(f"PRAGMA table_info({tbl})").fetchall()]
                        score_col = next((c for c in cols if "score" in c or "mastery" in c), None)
                        if "session_id" in cols and "concept_id" in cols and score_col:
                            s_rows = con.execute(
                                f"SELECT concept_id, {score_col} AS sc FROM {tbl} WHERE session_id = ?",
                                (sid,),
                            ).fetchall()
                            if s_rows:
                                saved_scores = {sr["concept_id"]: float(sr["sc"]) for sr in s_rows}
                                break
                    except Exception:
                        pass

            results.append({
                "session_id": sid,
                "submitted_at": d.get("submitted_at", "N/A"),
                "correct_count": correct_c or 0,
                "answered_count": answered_c or 0,
                "total_questions": total_q_saved,
                "score_10": round(float(score_val or 0.0), 2),
                "saved_scores": saved_scores,
                "details": details_list,
            })
        return results


def get_project_docs() -> list[str]:
    """Lấy danh sách các file Slide/Giáo trình trong thư mục docs/ và share_data/"""
    exts = (".pdf", ".pptx", ".ppt", ".docx", ".doc")
    search_dirs = [
        os.path.join(ROOT_DIR, "docs"),
        os.path.join(ROOT_DIR, "share_data"),
        os.path.join(ROOT_DIR, "materials"),
    ]
    found = []
    seen = set()
    for d in search_dirs:
        if os.path.exists(d):
            for root, _, files in os.walk(d):
                for f in sorted(files):
                    if f.lower().endswith(exts):
                        full_p = os.path.join(root, f)
                        if f.lower() not in seen:
                            seen.add(f.lower())
                            found.append(full_p)
    return found


def render_pdf_preview(file_path: str, height: int = 560):
    """Hiển thị trực tiếp file PDF dung lượng lớn không bị giới hạn Base64"""
    if hasattr(st, "pdf"):
        st.pdf(file_path, height=height)
    else:
        # Fallback nếu phiên bản Streamlit cũ hơn 1.49: trích xuất & hiển thị từng trang bằng PyMuPDF (fitz)
        try:
            import fitz
            doc = fitz.open(file_path)
            total_pages = len(doc)
            page_num = st.number_input(
                f"Chọn trang cần xem (Tổng {total_pages} trang):",
                min_value=1,
                max_value=total_pages,
                value=1,
                key=f"pdf_page_{os.path.basename(file_path)}",
            )
            page = doc[page_num - 1]
            pix = page.get_pixmap(dpi=150)
            st.image(pix.tobytes("png"), use_container_width=True)
        except Exception:
            st.warning(
                "Tệp giáo trình có dung lượng lớn nên trình duyệt chặn xem nhúng trực tiếp. "
                "Vui lòng nhấn nút **Tải xuống** ở thanh bên trái để mở bằng trình đọc PDF của máy."
            )

# ==============================================================================
# SIDEBAR: ĐIỀU KHIỂN, TÀI LIỆU & CHÚ THÍCH
# ==============================================================================
with st.sidebar:
    st.title("QLDA CNTT - GNN")
    st.caption("Hệ thống Hỗ trợ Học tập Thông minh bằng Graph Attention Network")
    st.divider()

    st.markdown("### Cấu hình bài kiểm tra")
    quiz_mode = st.radio(
        "Chọn quy mô đề:",
        options=[1, 2],
        format_func=lambda x: "Chẩn đoán nhanh (27 câu - Khuyên dùng)" if x == 1 else "Đánh giá chuyên sâu (54 câu)",
        index=0 if st.session_state.limit_per_concept == 1 else 1,
    )

    if st.button(f"Tạo đề chẩn đoán mới ({quiz_mode * 27} câu)", type="primary", use_container_width=True):
        load_new_quiz(limit_per_concept=quiz_mode)
        st.rerun()

    if st.button("Chạy Demo nhanh (Điền sẵn bài mẫu)", use_container_width=True):
        load_new_quiz(limit_per_concept=quiz_mode)
        demo_ans = build_demo_user_answers(st.session_state.quiz_questions)
        grade_res = db.grade_submission(demo_ans, session_id=st.session_state.session_id)
        gnn_scores = run_gnn_diagnosis(grade_res["details"])

        if st.session_state.session_id is not None:
            db.save_gnn_mastery_scores(st.session_state.session_id, gnn_scores)

        st.session_state.grade_result = grade_res
        st.session_state.answered_count = len(demo_ans)
        st.session_state.mastery_scores = gnn_scores
        st.session_state.loaded_history_details = None
        st.session_state.scroll_to_top_after_grade = True
        st.rerun()

    st.divider()
    st.markdown("### Kho Tài liệu Học tập")
    project_docs = get_project_docs()
    if not project_docs:
        st.caption("Chưa có file tài liệu trong thư mục `docs/`. Hãy chép các file PDF Slide và Giáo trình vào `docs/` để tra cứu tại đây.")
    else:
        doc_map = {os.path.basename(p): p for p in project_docs}
        selected_doc = st.selectbox(
            "Chọn Slide hoặc Giáo trình:",
            options=list(doc_map.keys()),
            key="sidebar_doc_select",
        )
        selected_doc_path = doc_map[selected_doc]

        with open(selected_doc_path, "rb") as f_doc:
            st.download_button(
                label=f"Tải xuống ({selected_doc})",
                data=f_doc.read(),
                file_name=selected_doc,
                use_container_width=True,
            )

        if selected_doc_path.lower().endswith(".pdf"):
            if st.toggle("Xem trực tiếp PDF trên màn hình", key="sidebar_pdf_toggle"):
                st.session_state.preview_pdf_path = selected_doc_path
            else:
                st.session_state.preview_pdf_path = None

    st.divider()
    st.markdown("### Chú thích Bản đồ Tri thức")
    st.markdown("🔴 **Đỏ (< 50%):** Lỗ hổng kiến thức")
    st.markdown("🟠 **Cam (50% - 69%):** Cần củng cố")
    st.markdown("🟢 **Xanh (≥ 70%):** Đã nắm vững")


# ==============================================================================
# GIAO DIỆN CHÍNH (4 TABS)
# ==============================================================================
st.header("Hệ thống Chẩn đoán Lỗ hổng Kiến thức & Gợi ý Lộ trình Học tập")

# Khung đọc PDF trực tiếp nếu người dùng bật công tắc ở Sidebar
if st.session_state.get("preview_pdf_path") and os.path.exists(st.session_state.preview_pdf_path):
    doc_title = os.path.basename(st.session_state.preview_pdf_path)
    with st.expander(f"Trình đọc tài liệu: {doc_title}", expanded=True):
        render_pdf_preview(st.session_state.preview_pdf_path, height=560)

tab1, tab2, tab3, tab4 = st.tabs([
    "1. Kiểm tra Chẩn đoán",
    "2. Bản đồ Tri thức (GNN Graph)",
    "3. Lộ trình Ôn tập & Tài liệu",
    "4. Lịch sử Học tập",
])

# ------------------------------------------------------------------------------
# TAB 1: LÀM BÀI KIỂM TRA CHẨN ĐOÁN (27 HOẶC 54 CÂU)
# ------------------------------------------------------------------------------
with tab1:
    if not st.session_state.quiz_questions:
        load_new_quiz(limit_per_concept=quiz_mode)

    questions = st.session_state.quiz_questions
    for q in questions:
        qid = q["question_id"]
        if qid not in st.session_state.quiz_option_orders:
            st.session_state.quiz_option_orders[qid] = randomize_option_order()
            st.session_state.pop(f"radio_{qid}", None)

    grade_details = {
        detail["question_id"]: detail
        for detail in (st.session_state.grade_result or {}).get("details", [])
    }

    min_required = min(10, len(questions))
    st.info(
        f"Bài kiểm tra gồm **{len(questions)} câu hỏi** phủ đều 27 chủ đề cốt lõi. "
        f"Bạn cần hoàn thành tối thiểu **{min_required} câu** để nộp bài; những câu bỏ trống mô hình **GAT** sẽ tự động suy luận thông qua đồ thị tri thức."
    )

    if st.session_state.grade_result:
        grade_res = st.session_state.grade_result
        total_q = len(questions)
        correct_c = grade_res["correct_count"]
        answered_c = st.session_state.answered_count
        skipped_c = total_q - answered_c
        wrong_c = answered_c - correct_c
        score_on_total = round((correct_c / total_q) * 10, 2) if total_q > 0 else 0.0

        st.success(
            f"Hoàn thành bài kiểm tra. Điểm quy đổi toàn đề: **{score_on_total}/10** — "
            f"Đúng: **{correct_c} câu** | Sai: **{wrong_c} câu** | Bỏ trống: **{skipped_c} câu**. "
            f"Chuyển sang **Tab 2** để xem Bản đồ tri thức và **Tab 3** để xem Lộ trình ôn tập."
        )

        with st.expander("Xem chi tiết dữ liệu đầu vào đã nạp vào mô hình GAT", expanded=True):
            wrong_list = [d["question_id"] for d in grade_res["details"] if not d["is_correct"]]
            correct_list = [d["question_id"] for d in grade_res["details"] if d["is_correct"]]
            answered_set = {d["question_id"] for d in grade_res["details"]}
            skipped_list = [q["question_id"] for q in questions if q["question_id"] not in answered_set]

            c_ok, c_err, c_skip = st.columns(3)
            c_ok.markdown(f"**Trả lời ĐÚNG (Nhãn `1.0`):**\n\n`{', '.join(correct_list) if correct_list else 'Không có'}`")
            c_err.markdown(f"**Trả lời SAI (Nhãn `0.0`):**\n\n`{', '.join(wrong_list) if wrong_list else 'Không có'}`")
            c_skip.markdown(f"**Bỏ trống (Nhãn ẩn `0.5`):**\n\n`{', '.join(skipped_list) if skipped_list else 'Không có'}`")

    with st.form("diagnostic_quiz_form"):
        user_answers = {}
        for idx, q in enumerate(questions, start=1):
            qid = q["question_id"]
            diff = q.get("difficulty", "Hiểu")
            option_order = st.session_state.quiz_option_orders[qid]
            display_to_original = dict(zip(["A", "B", "C", "D"], option_order))
            original_to_display = {
                original: display for display, original in display_to_original.items()
            }
            options = {
                display: escape_markdown(f"{display}. {q['option_' + original.lower()]}")
                for display, original in display_to_original.items()
            }
            result = grade_details.get(qid)
            correct_display = original_to_display.get(result["correct_answer"]) if result else None
            selected_display = original_to_display.get(result["chosen"]) if result else None

            if result:
                status = "[Đúng] " if result["is_correct"] else "[Sai] "
            elif st.session_state.grade_result is not None:
                status = "[Bỏ trống] "
            else:
                status = ""

            q_text_safe = escape_markdown(q["question_text"])
            st.markdown(f"**{status}Câu {idx} ({qid} - Mức độ: {diff}):** {q_text_safe}")

            def format_option(
                label,
                option_texts=options,
                correct=correct_display,
                selected=selected_display,
            ):
                if label == correct:
                    return f"✅ {option_texts[label]}"
                if label == selected:
                    return f"❌ {option_texts[label]}"
                return option_texts[label]

            choice = st.radio(
                label=f"Chọn đáp án cho {qid}",
                options=list(options),
                format_func=format_option,
                index=["A", "B", "C", "D"].index(selected_display) if selected_display else None,
                key=f"radio_{qid}",
                label_visibility="collapsed",
                disabled=st.session_state.grade_result is not None,
            )
            if choice:
                user_answers[qid] = display_to_original[choice]
            st.markdown("---")

        submitted = st.form_submit_button(
            "Nộp bài & Phân tích bằng AI (GAT)",
            type="primary",
            disabled=st.session_state.grade_result is not None,
        )

    if submitted:
        if len(user_answers) < min_required:
            st.warning(
                f"Bạn mới làm {len(user_answers)}/{len(questions)} câu. "
                f"Hãy hoàn thành tối thiểu **{min_required} câu** để hệ thống có đủ dữ liệu phân tích."
            )
        else:
            with st.spinner("Đang chấm điểm và chạy suy luận mạng nơ-ron đồ thị GAT..."):
                if st.session_state.session_id is None:
                    st.session_state.session_id = db.start_test_session(user_id=None)

                grade_res = db.grade_submission(user_answers, session_id=st.session_state.session_id)
                gnn_scores = run_gnn_diagnosis(grade_res["details"])

                if st.session_state.session_id is not None:
                    db.save_gnn_mastery_scores(st.session_state.session_id, gnn_scores)

                st.session_state.grade_result = grade_res
                st.session_state.answered_count = len(user_answers)
                st.session_state.mastery_scores = gnn_scores
                st.session_state.loaded_history_details = None
                st.session_state.scroll_to_top_after_grade = True
            st.rerun()

    if st.session_state.scroll_to_top_after_grade:
        components.html(
            """
            <script>
            window.setTimeout(() => {
                const main = window.parent.document.querySelector('[data-testid="stMain"]');
                if (main) main.scrollTo({ top: 0, behavior: "smooth" });
            }, 100);
            </script>
            """,
            height=0,
        )
        st.session_state.scroll_to_top_after_grade = False

# ------------------------------------------------------------------------------
# TAB 2: BẢN ĐỒ TRI THỨC CÁ NHÂN HÓA (PYVIS GRAPH)
# ------------------------------------------------------------------------------
with tab2:
    if st.session_state.mastery_scores is None:
        st.warning("Chưa có dữ liệu phân tích. Hãy hoàn thành bài kiểm tra ở **Tab 1** hoặc chọn **Chạy Demo nhanh** ở thanh bên trái.")
    else:
        scores = st.session_state.mastery_scores
        all_concepts = db.get_all_concepts()

        if st.session_state.loaded_history_details is not None:
            details_list = st.session_state.loaded_history_details
            active_questions = []
            total_q_count = st.session_state.loaded_history_total_q or len(details_list)
        else:
            details_list = (st.session_state.grade_result or {}).get("details", [])
            active_questions = st.session_state.quiz_questions
            total_q_count = len(st.session_state.quiz_questions) or len(details_list)

        st.markdown("#### Tổng quan Dữ liệu Bài làm đầu vào")
        if details_list:
            answered_q_count = len(details_list)
            skipped_q_count = max(0, total_q_count - answered_q_count)
            correct_q_count = sum(1 for d in details_list if d.get("is_correct"))
            wrong_q_count = answered_q_count - correct_q_count

            q_col1, q_col2, q_col3, q_col4, q_col5 = st.columns(5)
            q_col1.metric("Tổng số câu đề", f"{total_q_count} câu")
            q_col2.metric("Đã làm", f"{answered_q_count} câu")
            q_col3.metric("Chưa làm (0.5)", f"{skipped_q_count} câu")
            q_col4.metric("Làm đúng (1.0)", f"{correct_q_count} câu")
            q_col5.metric("Làm sai (0.0)", f"{wrong_q_count} câu")
        else:
            st.info("Lượt kiểm tra cũ này chỉ lưu điểm thành thạo 27 chủ đề (không có dữ liệu chi tiết từng câu trả lời).")

        st.divider()

        st.markdown("#### Kết quả Chẩn đoán 27 Chủ đề Kiến thức (GAT)")
        scored_concepts = [c for c in all_concepts if c["concept_id"] in scores]
        red_list = [c for c in scored_concepts if scores[c["concept_id"]] < 0.50]
        yellow_list = [c for c in scored_concepts if 0.50 <= scores[c["concept_id"]] < 0.70]
        green_list = [c for c in scored_concepts if scores[c["concept_id"]] >= 0.70]

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Tổng số chủ đề", len(all_concepts))
        col2.metric("🔴 Lỗ hổng (< 50%)", len(red_list))
        col3.metric("🟠 Cần củng cố (50-69%)", len(yellow_list))
        col4.metric("🟢 Nắm vững (≥ 70%)", len(green_list))

        chapters = ["Tất cả"] + sorted(list({c["chapter"] for c in all_concepts}))
        selected_chap = st.selectbox("Lọc hiển thị theo Chương / Bài học:", chapters)

        st.caption("Gợi ý: Rê chuột trực tiếp lên từng nút tròn trên đồ thị để xem thông tin chủ đề và trạng thái câu hỏi liên quan.")
        graph_html = build_knowledge_graph_html(
            scores, selected_chapter=selected_chap, grade_details=details_list
        )
        components.html(graph_html, height=620, scrolling=True)

        st.divider()
        st.subheader("Phân tích & Giải thích Chi tiết từng Chủ đề")
        explanations = generate_concept_explanations(
            scores, details_list, active_questions
        )

        filter_options = [
            f"🔴 Lỗ hổng kiến thức ({len(red_list)} chủ đề)",
            f"🟠 Cần củng cố ({len(yellow_list)} chủ đề)",
            f"🟢 Đã nắm vững ({len(green_list)} chủ đề)",
            f"Tất cả ({len(scored_concepts)} chủ đề)",
        ]

        selected_group = st.radio(
            "Chọn nhóm kiến thức cần xem giải thích:",
            options=filter_options,
            horizontal=True,
        )

        if selected_group.startswith("🔴"):
            target_concepts = red_list
            badge = "🔴"
            empty_msg = "Không có chủ đề nào bị hổng (< 50%)."
        elif selected_group.startswith("🟠"):
            target_concepts = yellow_list
            badge = "🟠"
            empty_msg = "Không có chủ đề nào ở mức Cần củng cố (50% - 69%)."
        elif selected_group.startswith("🟢"):
            target_concepts = green_list
            badge = "🟢"
            empty_msg = "Chưa có chủ đề nào đạt mức Nắm vững (≥ 70%)."
        else:
            target_concepts = scored_concepts
            badge = None
            empty_msg = "Chưa có dữ liệu phân tích."

        if not target_concepts:
            st.info(empty_msg)
        else:
            for c in target_concepts:
                cid = c["concept_id"]
                exp = explanations[cid]
                sc_val = exp["score"]
                icon = badge or ("🔴" if sc_val < 0.50 else "🟠" if sc_val < 0.70 else "🟢")

                with st.container(border=True):
                    st.markdown(
                        f"**{icon} {cid} · {exp['concept_name']}** — Thành thạo: **{sc_val * 100:.0f}%** · *{exp['chapter']}*"
                    )
                    for r in exp["reasons"]:
                        st.markdown(f"- {r}", unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# TAB 3: LỘ TRÌNH ÔN TẬP & CHỈ DẪN TÀI LIỆU
# ------------------------------------------------------------------------------
with tab3:
    if st.session_state.mastery_scores is None:
        st.warning("Chưa có dữ liệu chẩn đoán để đề xuất lộ trình.")
    else:
        scores = st.session_state.mastery_scores
        all_concepts = db.get_all_concepts()

        if st.session_state.loaded_history_details is not None:
            details_list = st.session_state.loaded_history_details
            active_questions = []
        else:
            details_list = (st.session_state.grade_result or {}).get("details", [])
            active_questions = st.session_state.quiz_questions

        explanations = generate_concept_explanations(scores, details_list, active_questions)

        st.subheader("Lộ trình Học tập Cá nhân hóa theo Thứ tự Tiên quyết")
        route_scope = st.radio(
            "Chọn phạm vi lộ trình ôn tập:",
            options=["RED_ONLY", "RED_AND_YELLOW"],
            format_func=lambda x: (
                "Tập trung khắc phục Lỗ hổng (< 50%) — Khuyên dùng"
                if x == "RED_ONLY"
                else "Ôn tập toàn diện (Lỗ hổng < 50% + Cần củng cố < 70%)"
            ),
            horizontal=True,
        )

        threshold = 0.50 if route_scope == "RED_ONLY" else 0.70
        target_concepts = [c for c in all_concepts if scores.get(c["concept_id"], 0.5) < threshold]

        if not target_concepts:
            st.success("Không phát hiện chủ đề nào cần ôn tập trong phạm vi đã chọn.")
        else:
            st.info(
                f"Lộ trình gồm **{len(target_concepts)} bước** được sắp xếp từ kiến thức nền tảng đến nâng cao. "
                f"Bạn có thể mở trực tiếp tệp Slide hoặc Giáo trình tương ứng tại mục Kho Tài liệu ở thanh bên trái."
            )

            all_q_rows = []
            db_path = os.path.join(ROOT_DIR, "data_pipeline", "database.db")
            if os.path.exists(db_path):
                with sqlite3.connect(db_path) as con:
                    con.row_factory = sqlite3.Row
                    all_q_rows = [dict(r) for r in con.execute("SELECT * FROM questions").fetchall()]
            q_by_id = {q["question_id"]: q for q in all_q_rows}

            edges_df = pd.read_csv(os.path.join(ROOT_DIR, "share_data", "edges.csv"))
            c_to_qids = (
                edges_df[edges_df["relation"] == "tests"]
                .groupby("target_id")["source_id"]
                .apply(list)
                .to_dict()
            )

            for step, c in enumerate(target_concepts, start=1):
                cid = c["concept_id"]
                sc = scores[cid]
                is_red = sc < 0.50
                status_text = "Lỗ hổng cần vá ngay" if is_red else "Cần củng cố thêm"
                exp_info = explanations.get(cid, {})
                ref_data = db.get_concept_references(cid, max_excerpt_chars=900)

                with st.container(border=True):
                    step_col, concept_col, score_col = st.columns([1, 5.2, 1.5])
                    with step_col:
                        st.caption("BƯỚC")
                        st.markdown(f"## {step:02d}")
                    with concept_col:
                        st.markdown(f"#### `{cid}` · {c['concept_name']} *({status_text})*")
                        st.caption(f"{c['chapter']}")
                    with score_col:
                        st.metric("Thành thạo", f"{sc:.0%}", delta="Mục tiêu ≥ 70%")

                    st.progress(min(1.0, max(0.0, float(sc))))

                    if exp_info.get("reasons"):
                        st.markdown("**Chẩn đoán & Mục tiêu bước này:**")
                        for r in exp_info["reasons"][:2]:
                            st.markdown(f"- {r}", unsafe_allow_html=True)

                    st.markdown("---")

                    # Chỉ dẫn rõ tên tệp và số trang trong Slide & Giáo trình
                    if ref_data:
                        slide_ref = ref_data.get("slide", {})
                        textbook_ref = ref_data.get("textbook", {})
                        slide_col, textbook_col = st.columns(2)

                        with slide_col:
                            st.markdown("**Vị trí trong Slide bài giảng**")
                            s_file = slide_ref.get("file")
                            s_pages = slide_ref.get("pages_label")
                            if s_file or s_pages:
                                st.info(f"Tệp: **`{s_file or 'Slide bài giảng'}`** — **Trang {s_pages or 'N/A'}**")
                            else:
                                st.caption("Chưa gắn chỉ mục slide cho chủ đề này.")

                        with textbook_col:
                            st.markdown("**Vị trí trong Giáo trình tham khảo**")
                            t_file = textbook_ref.get("file")
                            t_pages = textbook_ref.get("pages_label")
                            if t_file or t_pages:
                                st.info(f"Tệp: **`{t_file or 'Giáo trình QLDA CNTT'}`** — **Trang {t_pages or 'N/A'}**")
                            else:
                                st.caption("Chưa gắn chỉ mục giáo trình cho chủ đề này.")

                        all_excerpts = slide_ref.get("excerpts", []) + textbook_ref.get("excerpts", [])
                        if all_excerpts:
                            with st.expander("Đọc nhanh nội dung trọng tâm", expanded=is_red):
                                for ex in all_excerpts[:3]:
                                    st.markdown(f"**{ex.get('source', 'Tài liệu')} · Trang {ex.get('page', 'N/A')}:**")
                                    st.write(escape_markdown(ex.get("text", "")))
                    else:
                        st.caption("Chưa có dữ liệu tài liệu tham khảo cho chủ đề này.")

                    related_qids = c_to_qids.get(cid, [])
                    sample_qs = [q_by_id[qid] for qid in related_qids if qid in q_by_id]
                    if sample_qs:
                        with st.expander("Tự kiểm tra nhanh kiến thức bước này (Kèm đáp án)", expanded=False):
                            for sq in sample_qs[:2]:
                                st.markdown(f"**Câu hỏi ({sq['question_id']} - Mức độ: {sq.get('difficulty', 'Hiểu')}):** {escape_markdown(sq['question_text'])}")
                                for opt_key in ["A", "B", "C", "D"]:
                                    opt_val = escape_markdown(sq.get(f"option_{opt_key.lower()}", ""))
                                    if opt_key == sq.get("correct_answer"):
                                        st.markdown(f"- ✅ **{opt_key}. {opt_val}** *(Đáp án đúng)*")
                                    else:
                                        st.markdown(f"- {opt_key}. {opt_val}")
                                st.markdown("")

# ------------------------------------------------------------------------------
# TAB 4: LỊCH SỬ LÀM BÀI & THEO DÕI TIẾN BỘ
# ------------------------------------------------------------------------------
with tab4:
    st.subheader("Lịch sử các lần kiểm tra chẩn đoán")
    history = get_all_test_history()
    if not history:
        st.info("Chưa có lượt nộp bài nào được lưu trong hệ thống. Hãy hoàn thành bài kiểm tra ở **Tab 1** và nhấn Nộp bài.")
    else:
        st.success(f"Hệ thống đã ghi nhận **{len(history)} lượt làm bài**:")
        for idx, h in enumerate(history, start=1):
            attempt_no = len(history) - idx + 1
            saved_scores = h["saved_scores"]
            red_cnt = sum(1 for sc in saved_scores.values() if sc < 0.50)

            with st.container(border=True):
                c1, c2, c3, c4, c5 = st.columns([1.3, 2.7, 1.8, 1.8, 2.2])
                c1.markdown(f"**Lần #{attempt_no}**")
                c2.caption(f"Nộp lúc:\n{h['submitted_at']}")
                c3.metric(
                    "Số câu đúng",
                    f"{h['correct_count']}/{h['answered_count']} câu" if h["answered_count"] else f"{h['correct_count']} câu",
                )
                c4.metric("Điểm (trên câu làm)", f"{h['score_10']}/10")
                with c5:
                    if saved_scores:
                        st.caption(f"Phát hiện **{red_cnt}/27** lỗ hổng")
                        if st.button("Nạp lại kết quả này", key=f"load_hist_{h['session_id']}_{idx}"):
                            st.session_state.mastery_scores = saved_scores
                            st.session_state.loaded_history_details = h["details"] if h["details"] else []
                            st.session_state.loaded_history_total_q = h["total_questions"]
                            st.toast(f"Đã nạp Bản đồ tri thức & Lộ trình của Lần #{attempt_no}.")
                            st.rerun()
                    else:
                        st.caption("Chưa có dữ liệu đồ thị")