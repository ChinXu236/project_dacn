import os
import sys
import pandas as pd

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_DIR = os.path.join(ROOT_DIR, "model_gnn")
SHARE_DATA_DIR = os.path.join(ROOT_DIR, "share_data")

if MODEL_DIR not in sys.path:
    sys.path.insert(0, MODEL_DIR)

from model_gnn.inference import predict_mastery


def run_gnn_diagnosis(grade_details: list[dict]) -> dict[str, float]:
    """
    Chuyển đổi kết quả chấm điểm từng câu từ database.py sang đầu vào cho mô hình GAT.
    - Câu trả lời đúng -> 1.0
    - Câu trả lời sai -> 0.0
    - Các câu chưa làm -> mặc định 0.5 trong inference.py
    """
    student_answers = {}
    for item in grade_details:
        qid = item["question_id"]
        student_answers[qid] = 1.0 if item["is_correct"] else 0.0

    return predict_mastery(student_answers)


def build_demo_user_answers(questions: list[dict]) -> dict[str, str]:
    """
    Tạo bộ đáp án mẫu trực quan cho danh sách câu hỏi đang mở trên Tab 1:
    - Cố tình làm SAI các câu thuộc cụm kiến thức trọng tâm: C05 (WBS), C08 (CPM), C12 (EVM), C21 (Rủi ro)
    - BỎ TRỐNG 5 câu cuối để minh họa khả năng suy luận dữ liệu khuyết của GAT
    - Làm ĐÚNG các câu còn lại
    """
    edges_df = pd.read_csv(os.path.join(SHARE_DATA_DIR, "edges.csv"))
    questions_df = pd.read_csv(os.path.join(SHARE_DATA_DIR, "questions.csv"))

    test_edges = edges_df[edges_df["relation"] == "tests"]
    q_to_concepts = test_edges.groupby("source_id")["target_id"].apply(set).to_dict()
    q_to_correct = dict(zip(questions_df["question_id"], questions_df["correct_answer"]))

    weak_targets = {"C05", "C08", "C12", "C21"}
    demo_answers = {}

    # Bỏ trống 5 câu cuối cùng của đề để test khả năng suy luận câu chưa làm
    active_questions = questions[:-5] if len(questions) > 15 else questions

    for q in active_questions:
        qid = q["question_id"]
        correct_opt = q_to_correct.get(qid, "A")
        linked_concepts = q_to_concepts.get(qid, set())

        if linked_concepts & weak_targets:
            # Chọn 1 đáp án sai khác với đáp án đúng
            wrong_opt = next(opt for opt in ["A", "B", "C", "D"] if opt != correct_opt)
            demo_answers[qid] = wrong_opt
        else:
            demo_answers[qid] = correct_opt

    return demo_answers