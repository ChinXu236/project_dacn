import os
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from inference import GNNMasteryPredictor
from gnn_model import WeightedAverageBaseline

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "share_data")
MODEL_DIR = os.path.dirname(__file__)
TEST_FILE = os.path.join(MODEL_DIR, "real_and_persona_test.csv")

np.random.seed(42)


def parse_weak_concepts(weak_str, num_concepts=27):
    y_bin = np.zeros(num_concepts, dtype=int)
    if pd.isna(weak_str) or str(weak_str).strip() in ["", "NONE", "nan"]:
        return y_bin
    for cid in str(weak_str).split(";"):
        cid = cid.strip()
        if cid.startswith("C") and cid[1:].isdigit():
            idx = int(cid[1:]) - 1
            if 0 <= idx < num_concepts:
                y_bin[idx] = 1
    return y_bin


def predict_question_score_from_mastery(mastery_vec, q_id, test_edges_df, c_to_idx):
    """Tính xác suất làm đúng câu hỏi q_id dựa trên điểm thành thạo các concept liên quan"""
    links = test_edges_df[test_edges_df["source_id"] == q_id]
    if len(links) == 0:
        return float(np.mean(mastery_vec))
    weights = links["weight"].values
    weights = weights / np.sum(weights)
    c_indices = [c_to_idx[cid] for cid in links["target_id"]]
    return float(np.dot(mastery_vec[c_indices], weights))


def main():
    concepts_df = pd.read_csv(os.path.join(DATA_DIR, "concepts.csv"))
    questions_df = pd.read_csv(os.path.join(DATA_DIR, "questions.csv"))
    edges_df = pd.read_csv(os.path.join(DATA_DIR, "edges.csv"))
    test_df = pd.read_csv(TEST_FILE)

    q_cols = [f"Q{i:02d}" for i in range(1, len(questions_df) + 1)]
    c_cols = [f"C{i:02d}" for i in range(1, len(concepts_df) + 1)]
    c_to_idx = {cid: i for i, cid in enumerate(c_cols)}
    c_name_map = dict(zip(concepts_df["concept_id"], concepts_df["concept_name"]))
    test_edges_df = edges_df[edges_df["relation"] == "tests"]

    # Lọc hồ sơ hợp lệ
    valid_rows = [row for _, row in test_df.iterrows() if not np.all(row[q_cols].values.astype(float) == 0.5)]
    eval_df = pd.DataFrame(valid_rows).reset_index(drop=True)

    gat_predictor = GNNMasteryPredictor()
    baseline_model = WeightedAverageBaseline(len(q_cols), len(c_cols), test_edges_df)

    # ==========================================================================
    # PHẦN 1: ĐÁNH GIÁ CHẨN ĐOÁN LỖ HỔNG TRÊN 30 PERSONA CHUẨN (CÓ GROUND-TRUTH)
    # ==========================================================================
    persona_df = eval_df[eval_df["source_type"] == "Persona"]
    y_true_p, y_gat_p, y_base_p = [], [], []

    for _, row in persona_df.iterrows():
        ans_dict = {q: float(row[q]) for q in q_cols}
        ans_tensor = torch.tensor([[float(row[q]) for q in q_cols]], dtype=torch.float32)

        gat_scores = np.array([gat_predictor.predict(ans_dict)[c] for c in c_cols])
        with torch.no_grad():
            base_scores = baseline_model(ans_tensor).squeeze(0).numpy()

        y_true_p.append(parse_weak_concepts(row["known_weak_concepts"], len(c_cols)))
        y_gat_p.append((gat_scores < 0.5).astype(int))
        y_base_p.append((base_scores < 0.5).astype(int))

    yt_p, yg_p, yb_p = np.array(y_true_p).flatten(), np.array(y_gat_p).flatten(), np.array(y_base_p).flatten()

    persona_metrics_df = pd.DataFrame([
        {
            "Model": "Baseline (Weighted Avg)",
            "Accuracy": accuracy_score(yt_p, yb_p),
            "Precision": precision_score(yt_p, yb_p, zero_division=0),
            "Recall": recall_score(yt_p, yb_p, zero_division=0),
            "F1-Score": f1_score(yt_p, yb_p, zero_division=0),
        },
        {
            "Model": "GAT (Đề xuất chính)",
            "Accuracy": accuracy_score(yt_p, yg_p),
            "Precision": precision_score(yt_p, yg_p, zero_division=0),
            "Recall": recall_score(yt_p, yg_p, zero_division=0),
            "F1-Score": f1_score(yt_p, yg_p, zero_division=0),
        }
    ])

    print("=" * 78)
    print("BẢNG 1: ĐÁNH GIÁ CHẨN ĐOÁN LỖ HỔNG TRÊN 30 HỒ SƠ KỊCH BẢN (PERSONA)")
    print("=" * 78)
    print(persona_metrics_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # ==========================================================================
    # PHẦN 2: ĐÁNH GIÁ KHÁCH QUAN TRÊN 20 SINH VIÊN THẬT (MASKING 25% CÂU ĐÃ LÀM)
    # ==========================================================================
    real_df = eval_df[eval_df["source_type"] == "Real_Student"]
    masked_true, masked_gat_prob, masked_base_prob = [], [], []
    real_student_reports = []

    for _, row in real_df.iterrows():
        full_ans = {q: float(row[q]) for q in q_cols}
        answered_qs = [q for q, val in full_ans.items() if val != 0.5]

        # Che 5 câu (25% số câu đã làm) để kiểm tra khả năng suy luận của GNN
        masked_qs = np.random.choice(answered_qs, size=5, replace=False)
        partial_ans = full_ans.copy()
        for mq in masked_qs:
            partial_ans[mq] = 0.5

        gat_partial_scores = np.array([gat_predictor.predict(partial_ans)[c] for c in c_cols])
        partial_tensor = torch.tensor([[partial_ans[q] for q in q_cols]], dtype=torch.float32)
        with torch.no_grad():
            base_partial_scores = baseline_model(partial_tensor).squeeze(0).numpy()

        for mq in masked_qs:
            masked_true.append(int(full_ans[mq]))
            masked_gat_prob.append(predict_question_score_from_mastery(gat_partial_scores, mq, test_edges_df, c_to_idx))
            masked_base_prob.append(predict_question_score_from_mastery(base_partial_scores, mq, test_edges_df, c_to_idx))

        # Chẩn đoán toàn bài 20 câu cho báo cáo chi tiết
        full_gat_dict = gat_predictor.predict(full_ans)
        red_concepts = [f"{c} ({c_name_map[c]}: {full_gat_dict[c]:.2f})"
                        for c in c_cols if full_gat_dict[c] < 0.5]
        real_student_reports.append({
            "case_id": row["case_id"],
            "sinh_vien": row["description"],
            "tu_danh_gia_yeu": row["known_weak_concepts"],
            "so_node_do_GAT": len(red_concepts),
            "chi_tiet_node_do_GAT": " | ".join(red_concepts) if red_concepts else "Không có lỗ hổng (<0.5)"
        })

    mt = np.array(masked_true)
    mg_p = np.array(masked_gat_prob)
    mb_p = np.array(masked_base_prob)

    real_metrics_df = pd.DataFrame([
        {
            "Model": "Baseline (Weighted Avg)",
            "Accuracy": accuracy_score(mt, (mb_p >= 0.5).astype(int)),
            "F1-Score": f1_score(mt, (mb_p >= 0.5).astype(int), zero_division=0),
            "AUC-ROC": roc_auc_score(mt, mb_p),
        },
        {
            "Model": "GAT (Đề xuất chính)",
            "Accuracy": accuracy_score(mt, (mg_p >= 0.5).astype(int)),
            "F1-Score": f1_score(mt, (mg_p >= 0.5).astype(int), zero_division=0),
            "AUC-ROC": roc_auc_score(mt, mg_p),
        }
    ])

    print("\n" + "=" * 78)
    print("BẢNG 2: DỰ ĐOÁN NĂNG LỰC LÀM BÀI TRÊN 20 SINH VIÊN THẬT (MASKING 25% CÂU HỎI)")
    print("=" * 78)
    print(real_metrics_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("=" * 78)

    # Xuất các file báo cáo
    persona_metrics_df.to_csv(os.path.join(MODEL_DIR, "persona_metrics.csv"), index=False)
    real_metrics_df.to_csv(os.path.join(MODEL_DIR, "real_test_metrics.csv"), index=False)
    pd.DataFrame(real_student_reports).to_csv(
        os.path.join(MODEL_DIR, "real_students_diagnosis.csv"), index=False, encoding="utf-8-sig"
    )
    print("\nĐã cập nhật đầy đủ 3 file kết quả trong model_gnn/!")


if __name__ == "__main__":
    main()