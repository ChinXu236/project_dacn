import os
import pandas as pd
from inference import GNNMasteryPredictor

MODEL_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(MODEL_DIR, "..", "share_data")

def main():
    test_df = pd.read_csv(os.path.join(MODEL_DIR, "real_and_persona_test.csv"))
    concepts_df = pd.read_csv(os.path.join(DATA_DIR, "concepts.csv"))
    questions_df = pd.read_csv(os.path.join(DATA_DIR, "questions.csv"))

    # Lấy dữ liệu của sinh viên thật REAL_09 (Tuyết Nhi)
    student = test_df[test_df["case_id"] == "REAL_09"].iloc[0]
    q_cols = questions_df["question_id"].tolist()
    ans_dict = {q: float(student[q]) if q in student else 0.5 for q in q_cols}

    predictor = GNNMasteryPredictor()
    scores, (att_edge_index, att_weights) = predictor.predict(ans_dict, return_attention=True)

    # Trung bình trọng số Attention qua 4 heads
    mean_att = att_weights.mean(dim=1).cpu().numpy()
    edges = att_edge_index.cpu().numpy()

    num_q = len(questions_df)
    q_ids = questions_df["question_id"].tolist()
    c_ids = concepts_df["concept_id"].tolist()
    c_names = dict(zip(c_ids, concepts_df["concept_name"]))

    print(f"=== PHÂN TÍCH GIẢI THÍCH (EXPLAINABILITY) CHO {student['case_id']}: {student['description']} ===")
    print(f"Sinh viên tự đánh giá yếu: {student['known_weak_concepts']}\n")

    print("Top 10 liên kết có trọng số Attention cao nhất khi GAT suy luận:")
    att_records = []
    for idx in range(edges.shape[1]):
        u, v = edges[0, idx], edges[1, idx]
        if u != v:  # Bỏ qua self-loop
            u_name = q_ids[u] if u < num_q else c_ids[u - num_q]
            v_name = q_ids[v] if v < num_q else c_ids[v - num_q]
            # Chỉ lấy chiều truyền thông tin vào node Concept
            if v >= num_q:
                att_records.append((u_name, v_name, c_names[v_name], float(mean_att[idx])))

    att_records.sort(key=lambda x: x[3], reverse=True)
    for src, dst, cname, w in att_records[:10]:
        print(f"  {src} ---> {dst} ({cname:<45}) | Attention Weight: {w:.4f}")

if __name__ == "__main__":
    main()