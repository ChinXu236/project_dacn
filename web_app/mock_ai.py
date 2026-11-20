import os
import sys

# Thêm đường dẫn thư mục model_gnn để import được inference.py và gnn_model.py
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_DIR = os.path.join(ROOT_DIR, "model_gnn")
if MODEL_DIR not in sys.path:
    sys.path.insert(0, MODEL_DIR)

from model_gnn.inference import predict_mastery


def run_gnn_diagnosis(grade_details: list[dict]) -> dict[str, float]:
    """
    Chuyển đổi kết quả chấm điểm từng câu từ database.py sang đầu vào cho mô hình GAT.
    - Câu trả lời đúng -> 1.0
    - Câu trả lời sai -> 0.0
    - Các câu chưa làm (trong kho 120 câu) -> mặc định 0.5 trong inference.py
    """
    student_answers = {}
    for item in grade_details:
        qid = item["question_id"]
        student_answers[qid] = 1.0 if item["is_correct"] else 0.0

    return predict_mastery(student_answers)


def get_demo_mastery_scores() -> dict[str, float]:
    """Dữ liệu mô phỏng nhanh cho chế độ Demo trên giao diện"""
    demo_answers = {
        "Q01": 1.0, "Q02": 1.0, "Q03": 1.0, "Q04": 1.0,
        "Q05": 0.0, "Q06": 1.0, "Q07": 0.0, "Q08": 0.0,
        "Q09": 1.0, "Q10": 1.0, "Q11": 0.0, "Q12": 0.0,
        "Q13": 1.0, "Q14": 1.0, "Q15": 1.0, "Q16": 1.0,
        "Q17": 1.0, "Q18": 1.0, "Q19": 1.0, "Q20": 1.0,
        "Q21": 0.0, "Q22": 0.0, "Q23": 1.0, "Q24": 1.0,
        "Q25": 1.0, "Q26": 1.0, "Q27": 1.0,
    }
    return predict_mastery(demo_answers)