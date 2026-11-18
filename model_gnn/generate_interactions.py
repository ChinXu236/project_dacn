import os
import numpy as np
import pandas as pd
import networkx as nx

# ==============================================================================
# CẤU HÌNH THAM SỐ MÔ PHỎNG IRT & ĐỒ THỊ TRI THỨC
# ==============================================================================
NUM_STUDENTS = 2000          # Số lượng sinh viên ảo cần sinh (Train + Val)
UNANSWERED_RATE = 0.40       # Tỷ lệ câu hỏi chưa làm (40% câu = 0.5, tương đương bài test 18 câu/30 câu)
RANDOM_SEED = 42

# Đường dẫn tương đối từ thư mục gốc project_dacn
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "share_data")
OUTPUT_DIR = os.path.dirname(__file__)

np.random.seed(RANDOM_SEED)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def load_graph_data():
    concepts_df = pd.read_csv(os.path.join(DATA_DIR, "concepts.csv"))
    questions_df = pd.read_csv(os.path.join(DATA_DIR, "questions.csv"))
    edges_df = pd.read_csv(os.path.join(DATA_DIR, "edges.csv"))
    return concepts_df, questions_df, edges_df


def build_prerequisite_order(concepts_df, edges_df):
    """Sắp xếp thứ tự học tập (Topological Sort) dựa trên cạnh Concept -> Concept"""
    G = nx.DiGraph()
    for cid in concepts_df["concept_id"]:
        G.add_node(cid)

    prereq_edges = edges_df[edges_df["relation"] == "prerequisite"]
    for _, row in prereq_edges.iterrows():
        G.add_edge(row["source_id"], row["target_id"])

    # Lấy thứ tự topo để tính năng lực từ kiến thức nền -> kiến thức nâng cao
    topo_order = list(nx.topological_sort(G))
    return G, topo_order


def simulate_student_mastery(num_students, concepts_df, prereq_graph, topo_order):
    """
    Bước 1 của IRT: Sinh năng lực ẩn (Theta) và điểm thành thạo thực tế (Mastery 0.0 - 1.0)
    có tính đến sự phụ thuộc tiên quyết trên đồ thị.
    """
    concept_ids = concepts_df["concept_id"].tolist()
    num_concepts = len(concept_ids)

    # Năng lực tổng quát của từng sinh viên (phân phối chuẩn N(0, 1))
    general_ability = np.random.normal(0.0, 1.0, size=(num_students, 1))
    # Năng lực riêng biệt trên từng concept
    specific_ability = np.random.normal(0.0, 0.7, size=(num_students, num_concepts))
    raw_theta = 0.7 * general_ability + 0.3 * specific_ability  # Ma trận (N, 27)

    cid_to_idx = {cid: idx for idx, cid in enumerate(concept_ids)}
    theta_propagated = np.zeros_like(raw_theta)

    # Lan truyền năng lực theo thứ tự tiên quyết từ C01 -> C27
    for cid in topo_order:
        idx = cid_to_idx[cid]
        preds = list(prereq_graph.predecessors(cid))
        if len(preds) == 0:
            theta_propagated[:, idx] = raw_theta[:, idx]
        else:
            pred_indices = [cid_to_idx[p] for p in preds]
            avg_pred_theta = np.mean(theta_propagated[:, pred_indices], axis=1)
            # 60% năng lực tự thân + 40% nền tảng từ kiến thức tiên quyết
            theta_propagated[:, idx] = 0.6 * raw_theta[:, idx] + 0.4 * avg_pred_theta

    # Quy đổi năng lực sang điểm thành thạo thực tế (Ground-truth Mastery Score trong [0, 1])
    mastery_scores = sigmoid(theta_propagated)
    return theta_propagated, mastery_scores


def simulate_question_responses(theta_propagated, concepts_df, questions_df, edges_df):
    """
    Bước 2 của IRT (2PL-IRT): Tính xác suất làm đúng từng câu hỏi dựa trên
    năng lực của sinh viên ở các concept liên quan và độ khó câu hỏi.
    """
    num_students = theta_propagated.shape[0]
    concept_ids = concepts_df["concept_id"].tolist()
    question_ids = questions_df["question_id"].tolist()
    cid_to_idx = {cid: idx for idx, cid in enumerate(concept_ids)}

    # Ánh xạ độ khó câu hỏi sang tham số b_q của mô hình IRT
    difficulty_map = {
        "Nhớ": -0.75,
        "Hiểu": 0.0,
        "Vận dụng": 0.75
    }

    test_edges = edges_df[edges_df["relation"] == "tests"]
    interactions = np.zeros((num_students, len(question_ids)), dtype=np.float32)

    for q_idx, q_row in questions_df.iterrows():
        qid = q_row["question_id"]
        b_q = difficulty_map.get(q_row["difficulty"], 0.0)
        a_q = 1.5  # Tham số độ phân biệt câu hỏi (Discrimination)

        # Lấy tất cả concept mà câu hỏi này kiểm tra kèm trọng số
        q_links = test_edges[test_edges["source_id"] == qid]
        if len(q_links) == 0:
            effective_theta = np.mean(theta_propagated, axis=1)
        else:
            weights = q_links["weight"].values
            weights = weights / np.sum(weights)  # Chuẩn hóa tổng trọng số = 1
            c_indices = [cid_to_idx[cid] for cid in q_links["target_id"]]
            effective_theta = np.dot(theta_propagated[:, c_indices], weights)

        # Hàm xác suất 2PL-IRT
        prob_correct = sigmoid(a_q * (effective_theta - b_q))

        # Lấy mẫu nhị phân: 1 = Đúng, 0 = Sai
        responses = (np.random.rand(num_students) < prob_correct).astype(np.float32)

        # Giả lập bài test rút gọn (Diagnostic Test): 40% câu chưa làm -> gán 0.5
        unanswered_mask = np.random.rand(num_students) < UNANSWERED_RATE
        responses[unanswered_mask] = 0.5

        interactions[:, q_idx] = responses

    return interactions


def main():
    print("1. Đang tải dữ liệu đồ thị từ share_data/...")
    concepts_df, questions_df, edges_df = load_graph_data()

    print("2. Đang phân tích cấu trúc tiên quyết (Concept -> Concept)...")
    prereq_graph, topo_order = build_prerequisite_order(concepts_df, edges_df)

    print(f"3. Đang mô phỏng năng lực IRT cho {NUM_STUDENTS} sinh viên ảo...")
    theta_prop, mastery_scores = simulate_student_mastery(
        NUM_STUDENTS, concepts_df, prereq_graph, topo_order
    )

    print("4. Đang sinh lịch sử làm bài trắc nghiệm (0 = Sai, 0.5 = Chưa làm, 1 = Đúng)...")
    interactions = simulate_question_responses(
        theta_prop, concepts_df, questions_df, edges_df
    )

    # Lưu kết quả ra 2 file CSV trong thư mục model_gnn/
    student_ids = [f"SV_{i+1:04d}" for i in range(NUM_STUDENTS)]

    df_interactions = pd.DataFrame(interactions, columns=questions_df["question_id"].tolist())
    df_interactions.insert(0, "student_id", student_ids)
    inter_path = os.path.join(OUTPUT_DIR, "simulated_interactions.csv")
    df_interactions.to_csv(inter_path, index=False)

    df_mastery = pd.DataFrame(np.round(mastery_scores, 4), columns=concepts_df["concept_id"].tolist())
    df_mastery.insert(0, "student_id", student_ids)
    mastery_path = os.path.join(OUTPUT_DIR, "simulated_mastery_labels.csv")
    df_mastery.to_csv(mastery_path, index=False)

    print("\n--- HOÀN TẤT SINH DỮ LIỆU IRT ---")
    print(f"- File đầu vào GNN (X): {inter_path} | Kích thước: {df_interactions.shape}")
    print(f"- File nhãn đích GNN (Y): {mastery_path} | Kích thước: {df_mastery.shape}")
    print(f"- Tỷ lệ câu trả lời: Đúng={(interactions==1).mean():.1%}, Sai={(interactions==0).mean():.1%}, Chưa làm={(interactions==0.5).mean():.1%}")
    print(f"- Tỷ lệ concept bị hổng (Mastery < 0.5): {(mastery_scores < 0.5).mean():.1%}")


if __name__ == "__main__":
    main()