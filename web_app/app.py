import os
import random
import sys
import streamlit as st
import streamlit.components.v1 as components

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import data_pipeline.database as db
from web_app.mock_ai import run_gnn_diagnosis, get_demo_mastery_scores
from web_app.graph_viz import build_knowledge_graph_html

st.set_page_config(
    page_title="Hệ thống Chẩn đoán Lỗ hổng Kiến thức QLDA CNTT",
    page_icon="🕸️",
    layout="wide",
)

# Khởi tạo Session State
if "user_id" not in st.session_state:
    st.session_state.user_id = None
if "username" not in st.session_state:
    st.session_state.username = "Khách (Guest)"
if "quiz_questions" not in st.session_state:
    st.session_state.quiz_questions = []
if "quiz_option_orders" not in st.session_state:
    st.session_state.quiz_option_orders = {}
if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "grade_result" not in st.session_state:
    st.session_state.grade_result = None
if "mastery_scores" not in st.session_state:
    st.session_state.mastery_scores = None
if "scroll_to_top_after_grade" not in st.session_state:
    st.session_state.scroll_to_top_after_grade = False


def randomize_option_order():
    order = random.sample(["A", "B", "C", "D"], k=4)
    if order == ["A", "B", "C", "D"]:
        order[0], order[1] = order[1], order[0]
    return order


def load_new_quiz():
    for question in st.session_state.quiz_questions:
        st.session_state.pop(f"radio_{question['question_id']}", None)

    st.session_state.quiz_questions = db.get_diagnostic_test(limit_per_concept=2, include_answer=False)
    st.session_state.quiz_option_orders = {
        question["question_id"]: randomize_option_order()
        for question in st.session_state.quiz_questions
    }
    st.session_state.session_id = db.start_test_session(user_id=st.session_state.user_id)
    st.session_state.grade_result = None
    st.session_state.mastery_scores = None
    st.session_state.scroll_to_top_after_grade = False


# ==============================================================================
# SIDEBAR: TÀI KHOẢN & ĐIỀU KHIỂN
# ==============================================================================
with st.sidebar:
    st.title("🎓 QLDA CNTT - GNN")
    st.caption("Hệ thống Hỗ trợ Học tập Thông minh bằng Graph Attention Network")
    st.divider()

    # Khu vực Đăng nhập / Đăng ký nhanh
    if st.session_state.user_id is None:
        with st.expander("🔐 Đăng nhập / Đăng ký lưu lịch sử", expanded=False):
            auth_mode = st.radio("Chế độ:", ["Đăng nhập", "Đăng ký"], horizontal=True)
            uname = st.text_input("Tên đăng nhập", key="auth_user")
            pwd = st.text_input("Mật khẩu", type="password", key="auth_pass")
            if st.button("Xác nhận", use_container_width=True):
                if auth_mode == "Đăng nhập":
                    uid = db.authenticate_user(uname, pwd)
                    if uid:
                        st.session_state.user_id = uid
                        st.session_state.username = uname
                        st.success(f"Xin chào {uname}!")
                        st.rerun()
                    else:
                        st.error("Sai tài khoản hoặc mật khẩu.")
                else:
                    try:
                        uid = db.register_user(uname, pwd)
                        st.session_state.user_id = uid
                        st.session_state.username = uname
                        st.success("Đăng ký thành công!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Không thể đăng ký: {e}")
    else:
        st.success(f"👤 Sinh viên: **{st.session_state.username}**")
        if st.button("Đăng xuất", use_container_width=True):
            st.session_state.user_id = None
            st.session_state.username = "Khách (Guest)"
            st.rerun()

    st.divider()
    if st.button("📝 Tạo đề chẩn đoán mới (54 câu)", type="primary", use_container_width=True):
        load_new_quiz()
        st.rerun()

    if st.button("⚡ Chạy Demo nhanh (Dữ liệu mẫu)", use_container_width=True):
        st.session_state.mastery_scores = get_demo_mastery_scores()
        st.session_state.grade_result = None
        st.toast("Đã nạp kết quả chẩn đoán mẫu từ mô hình GAT!")

    st.divider()
    st.markdown("### 🎨 Chú thích Bản đồ Tri thức")
    st.markdown("🔴 **Đỏ (< 50%):** Lỗ hổng kiến thức")
    st.markdown("🟠 **Vàng (50% - 69%):** Cần củng cố")
    st.markdown("🟢 **Xanh (≥ 70%):** Đã nắm vững")


# ==============================================================================
# GIAO DIỆN CHÍNH (3 TABS)
# ==============================================================================
st.header("🕸️ Hệ thống Chẩn đoán Lỗ hổng Kiến thức & Gợi ý Lộ trình Học tập")

tab1, tab2, tab3 = st.tabs([
    "1️⃣ Kiểm tra Chẩn đoán",
    "2️⃣ Bản đồ Tri thức (GNN Graph)",
    "3️⃣ Lộ trình Ôn tập & Slide",
])

# ------------------------------------------------------------------------------
# TAB 1: LÀM BÀI KIỂM TRA CHẨN ĐOÁN 54 CÂU
# ------------------------------------------------------------------------------
with tab1:
    if not st.session_state.quiz_questions:
        load_new_quiz()

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
    st.info(f"Bài kiểm tra gồm **{len(questions)} câu hỏi** phủ đều 27 chủ đề cốt lõi của môn Quản lý dự án CNTT.")
    if st.session_state.grade_result:
        grade_res = st.session_state.grade_result
        st.success(
            f"🎉 Hoàn thành! Điểm số của bạn: **{grade_res['score_10']}/10** "
            f"({grade_res['correct_count']}/{len(questions)} câu đúng)."
        )

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
                display: f"{display}. {q['option_' + original.lower()]}"
                for display, original in display_to_original.items()
            }
            result = grade_details.get(qid)
            correct_display = original_to_display.get(result["correct_answer"]) if result else None
            selected_display = original_to_display.get(result["chosen"]) if result else None
            status = ("✅ " if result["is_correct"] else "❌ ") if result else ""
            st.markdown(
                f"**{status}Câu {idx} ({qid} - Mức độ: {diff}):** {q['question_text']}"
            )

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
                disabled=result is not None,
            )
            if choice:
                user_answers[qid] = display_to_original[choice]
            st.markdown("---")

        submitted = st.form_submit_button(
            "✅ Nộp bài & Phân tích bằng AI (GAT)",
            type="primary",
            disabled=st.session_state.grade_result is not None,
        )

    if submitted:
        if len(user_answers) < len(questions):
            st.warning(f"Bạn mới làm {len(user_answers)}/{len(questions)} câu. Hãy hoàn thành tất cả các câu trước khi nộp bài nhé!")
        else:
            with st.spinner("Đang chấm điểm và chạy suy luận mạng nơ-ron đồ thị GAT..."):
                grade_res = db.grade_submission(user_answers, session_id=st.session_state.session_id)
                gnn_scores = run_gnn_diagnosis(grade_res["details"])

                if st.session_state.session_id is not None:
                    db.save_gnn_mastery_scores(st.session_state.session_id, gnn_scores)

                st.session_state.grade_result = grade_res
                st.session_state.mastery_scores = gnn_scores
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
        st.warning("Chưa có dữ liệu phân tích. Hãy hoàn thành bài kiểm tra ở **Tab 1** hoặc bấm **Chạy Demo nhanh** ở thanh bên trái.")
    else:
        scores = st.session_state.mastery_scores
        all_concepts = db.get_all_concepts()

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
        selected_chap = st.selectbox("🔍 Lọc hiển thị theo Chương / Bài học:", chapters)

        graph_html = build_knowledge_graph_html(scores, selected_chapter=selected_chap)
        components.html(graph_html, height=600, scrolling=True)

# ------------------------------------------------------------------------------
# TAB 3: LỘ TRÌNH ÔN TẬP & TRÍCH XUẤT TÀI LIỆU SLIDE / GIÁO TRÌNH
# ------------------------------------------------------------------------------
with tab3:
    if st.session_state.mastery_scores is None:
        st.warning("Chưa có dữ liệu chẩn đoán để đề xuất lộ trình.")
    else:
        scores = st.session_state.mastery_scores
        all_concepts = db.get_all_concepts()

        # Lọc các concept bị yếu (< 0.50) và sắp xếp theo trình tự tiên quyết C01 -> C27
        weak_concepts = [c for c in all_concepts if scores.get(c["concept_id"], 0.5) < 0.50]

        if not weak_concepts:
            st.balloons()
            st.success("🌟 Tuyệt vời! Mô hình GAT không phát hiện lỗ hổng kiến thức nào (< 50%)!")
        else:
            st.subheader("🧭 Lộ trình học tập cá nhân hóa")
            st.caption("Các bước được sắp xếp theo thứ tự tiên quyết — hãy bắt đầu từ bước đầu tiên.")
            st.info(f"📌 Bạn có **{len(weak_concepts)} chủ đề** cần củng cố. Học lần lượt và xem tài liệu gợi ý trong từng bước nhé!")

            for step, c in enumerate(weak_concepts, start=1):
                cid = c["concept_id"]
                sc = scores[cid]
                ref_data = db.get_concept_references(cid, max_excerpt_chars=500)

                with st.container(border=True):
                    step_col, concept_col, score_col = st.columns([1, 5, 1.5])
                    with step_col:
                        st.caption("BƯỚC")
                        st.markdown(f"### {step:02d}")
                    with concept_col:
                        st.markdown(f"#### 🔴 {cid} · {c['concept_name']}")
                        st.caption(f"📚 {c['chapter']}")
                    with score_col:
                        st.metric("Thành thạo", f"{sc:.0%}")

                    st.progress(float(sc))

                    if ref_data:
                        slide_ref = ref_data.get("slide", {})
                        textbook_ref = ref_data.get("textbook", {})
                        slide_col, textbook_col = st.columns(2)
                        with slide_col:
                            st.markdown("**📑 Slide bài giảng**")
                            if slide_ref.get("file") or slide_ref.get("pages_label"):
                                st.caption(
                                    f"{slide_ref.get('file') or 'Tài liệu chưa gắn tên'} · "
                                    f"Trang {slide_ref.get('pages_label') or 'N/A'}"
                                )
                            else:
                                st.caption("Chưa có slide tham khảo.")
                        with textbook_col:
                            st.markdown("**📖 Giáo trình tham khảo**")
                            if textbook_ref.get("file") or textbook_ref.get("pages_label"):
                                st.caption(
                                    f"{textbook_ref.get('file') or 'Tài liệu chưa gắn tên'} · "
                                    f"Trang {textbook_ref.get('pages_label') or 'N/A'}"
                                )
                            else:
                                st.caption("Chưa có giáo trình tham khảo.")

                        excerpts = slide_ref.get("excerpts", [])
                        if excerpts:
                            with st.expander("💡 Xem nội dung trọng tâm", expanded=False):
                                for ex in excerpts[:2]:
                                    st.markdown(
                                        f"**{ex.get('source', 'Tài liệu')} · Trang {ex.get('page', 'N/A')}**"
                                    )
                                    st.write(ex.get("text", ""))
                    else:
                        st.caption("Chưa có tài liệu tham khảo cho chủ đề này.")
