import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, f1_score, roc_auc_score
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from gnn_model import WeightedAverageBaseline, KnowledgeGCN, KnowledgeGAT

# ==============================================================================
# CẤU HÌNH HUẤN LUYỆN
# ==============================================================================
BATCH_SIZE = 32
EPOCHS = 80
LEARNING_RATE = 0.005
RANDOM_SEED = 42

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "share_data")
MODEL_DIR = os.path.dirname(__file__)

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_graph_components():
    """Đọc CSV và xây dựng cấu trúc đồ thị chuẩn cho PyTorch Geometric"""
    concepts_df = pd.read_csv(os.path.join(DATA_DIR, "concepts.csv"))
    questions_df = pd.read_csv(os.path.join(DATA_DIR, "questions.csv"))
    edges_df = pd.read_csv(os.path.join(DATA_DIR, "edges.csv"))

    num_q = len(questions_df)
    num_c = len(concepts_df)

    # 1. Trích xuất đặc trưng văn bản (TF-IDF 16 chiều) từ mô tả Concept & nội dung Câu hỏi
    tfidf = TfidfVectorizer(max_features=16)
    all_texts = (
        (questions_df["question_text"] + " " + questions_df["difficulty"]).tolist()
        + (concepts_df["concept_name"] + " " + concepts_df["description"]).tolist()
    )
    text_features = tfidf.fit_transform(all_texts).toarray()  # (num_q + num_c, 16)

    # 2. Ánh xạ chỉ số node: Question [0 .. num_q-1], Concept [num_q .. num_q+num_c-1]
    q_map = {qid: i for i, qid in enumerate(questions_df["question_id"])}
    c_map = {cid: i + num_q for i, cid in enumerate(concepts_df["concept_id"])}

    src_list, dst_list, weight_list = [], [], []
    for _, row in edges_df.iterrows():
        s_id, t_id, rel, w = row["source_id"], row["target_id"], row["relation"], float(row["weight"])
        if rel == "tests":
            u, v = q_map[s_id], c_map[t_id]
            # Thêm cạnh 2 chiều giữa Question và Concept để tin nhắn truyền qua lại
            src_list.extend([u, v])
            dst_list.extend([v, u])
            weight_list.extend([w, w])
        elif rel == "prerequisite":
            u, v = c_map[s_id], c_map[t_id]
            # Cạnh tiên quyết giữa Concept -> Concept (kèm chiều ngược trọng số nhẹ để suy luận lỗ hổng gốc)
            src_list.extend([u, v])
            dst_list.extend([v, u])
            weight_list.extend([w, 0.5 * w])

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    edge_weight = torch.tensor(weight_list, dtype=torch.float32)

    return concepts_df, questions_df, edges_df, text_features, edge_index, edge_weight


def create_pyg_dataset(interactions_np, mastery_np, text_features, edge_index, edge_weight, num_q, num_c):
    """Tạo danh sách các đối tượng PyG Data cho từng sinh viên"""
    dataset = []
    num_nodes = num_q + num_c

    # Node type indicator: [1, 0] cho Question, [0, 1] cho Concept
    node_type = np.zeros((num_nodes, 2), dtype=np.float32)
    node_type[:num_q, 0] = 1.0
    node_type[num_q:, 1] = 1.0

    for i in range(len(interactions_np)):
        # Feature cột đầu tiên: kết quả làm bài ở node Question; node Concept khởi tạo = 0.5 (chưa biết)
        status_feat = np.full((num_nodes, 1), 0.5, dtype=np.float32)
        status_feat[:num_q, 0] = interactions_np[i]

        # Ghép: [trạng thái làm bài (1), loại node (2), đặc trưng ngữ nghĩa TF-IDF (16)] -> 19 chiều
        x_np = np.hstack([status_feat, node_type, text_features]).astype(np.float32)
        data = Data(
            x=torch.tensor(x_np, dtype=torch.float32),
            edge_index=edge_index,
            edge_weight=edge_weight,
            y=torch.tensor(mastery_np[i], dtype=torch.float32)
        )
        dataset.append(data)
    return dataset


def evaluate_predictions(y_true, y_pred):
    """Tính các chỉ số hồi quy (RMSE, MAE) và phân loại lỗ hổng kiến thức (< 0.5: Accuracy, F1, AUC)"""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)

    # Nhãn nhị phân: 1 = Bị hổng kiến thức (Mastery < 0.5), 0 = Đạt (>= 0.5)
    y_true_bin = (y_true < 0.5).astype(int).flatten()
    y_pred_bin = (y_pred < 0.5).astype(int).flatten()
    prob_gap = (1.0 - y_pred).flatten()  # Xác suất bị hổng kiến thức

    acc = accuracy_score(y_true_bin, y_pred_bin)
    f1 = f1_score(y_true_bin, y_pred_bin)
    auc = roc_auc_score(y_true_bin, prob_gap)
    return {"RMSE": rmse, "MAE": mae, "Accuracy": acc, "F1-Score": f1, "AUC-ROC": auc}


def train_and_eval_gnn(model, train_loader, val_loader, num_c, use_edge_weight=False, model_name="GNN"):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(1, EPOCHS + 1):
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            if use_edge_weight:
                out = model(batch.x, batch.edge_index, edge_weight=batch.edge_weight)
            else:
                out = model(batch.x, batch.edge_index)

            # Reshape đầu ra theo từng sinh viên trong batch: (batch_size, num_c)
            out_reshaped = out.view(-1, num_c)
            y_reshaped = batch.y.view(-1, num_c)
            loss = criterion(out_reshaped, y_reshaped)
            loss.backward()
            optimizer.step()

        # Đánh giá trên tập Validation
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                if use_edge_weight:
                    out = model(batch.x, batch.edge_index, edge_weight=batch.edge_weight)
                else:
                    out = model(batch.x, batch.edge_index)
                val_preds.append(out.view(-1, num_c).cpu().numpy())
                val_targets.append(batch.y.view(-1, num_c).cpu().numpy())

        val_preds = np.vstack(val_preds)
        val_targets = np.vstack(val_targets)
        val_mse = mean_squared_error(val_targets, val_preds)

        if val_mse < best_val_loss:
            best_val_loss = val_mse
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0 or epoch == EPOCHS:
            print(f"[{model_name}] Epoch {epoch:02d}/{EPOCHS} | Val RMSE: {np.sqrt(val_mse):.4f}")

    model.load_state_dict(best_state)
    return model, val_preds, val_targets


def main():
    print("1. Đang nạp dữ liệu đồ thị và lịch sử tương tác IRT...")
    concepts_df, questions_df, edges_df, text_features, edge_index, edge_weight = build_graph_components()
    num_q, num_c = len(questions_df), len(concepts_df)

    df_inter = pd.read_csv(os.path.join(MODEL_DIR, "simulated_interactions.csv"))
    df_mastery = pd.read_csv(os.path.join(MODEL_DIR, "simulated_mastery_labels.csv"))

    X_inter = df_inter.drop(columns=["student_id"]).values.astype(np.float32)
    Y_mastery = df_mastery.drop(columns=["student_id"]).values.astype(np.float32)

    # Chia Train (80% = 1.600 SV) và Validation (20% = 400 SV) theo student_id
    idx_train, idx_val = train_test_split(np.arange(len(X_inter)), test_size=0.2, random_state=RANDOM_SEED)

    # --- ĐÁNH GIÁ 1: BASELINE (TRUNG BÌNH CỘNG CÓ TRỌNG SỐ) ---
    print("\n2. Đang chạy mô hình Baseline (Weighted Average)...")
    test_edges_df = edges_df[edges_df["relation"] == "tests"]
    baseline_model = WeightedAverageBaseline(num_q, num_c, test_edges_df)
    with torch.no_grad():
        base_preds = baseline_model(torch.tensor(X_inter[idx_val])).numpy()
    metrics_base = evaluate_predictions(Y_mastery[idx_val], base_preds)

    # Chuẩn bị PyG Dataset cho GCN và GAT
    train_dataset = create_pyg_dataset(X_inter[idx_train], Y_mastery[idx_train], text_features, edge_index, edge_weight, num_q, num_c)
    val_dataset = create_pyg_dataset(X_inter[idx_val], Y_mastery[idx_val], text_features, edge_index, edge_weight, num_q, num_c)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    in_channels = train_dataset[0].x.shape[1]  # 19 chiều

    # --- ĐÁNH GIÁ 2: MÔ HÌNH GCN ---
    print("\n3. Đang huấn luyện mô hình GCN...")
    gcn_model = KnowledgeGCN(in_channels=in_channels, hidden_channels=32, num_questions=num_q, num_concepts=num_c)
    gcn_model, gcn_preds, val_targets = train_and_eval_gnn(
        gcn_model, train_loader, val_loader, num_c, use_edge_weight=True, model_name="GCN"
    )
    metrics_gcn = evaluate_predictions(val_targets, gcn_preds)

    # --- ĐÁNH GIÁ 3: MÔ HÌNH GAT (ĐỀ XUẤT CHÍNH) ---
    print("\n4. Đang huấn luyện mô hình GAT (Multi-head Attention)...")
    gat_model = KnowledgeGAT(in_channels=in_channels, hidden_channels=16, heads=4, num_questions=num_q, num_concepts=num_c)
    gat_model, gat_preds, _ = train_and_eval_gnn(
        gat_model, train_loader, val_loader, num_c, use_edge_weight=False, model_name="GAT"
    )
    metrics_gat = evaluate_predictions(val_targets, gat_preds)

    # --- TỔNG HỢP BẢNG SO SÁNH & LƯU MODEL ---
    results_df = pd.DataFrame([
        {"Model": "Baseline (Weighted Avg)", **metrics_base},
        {"Model": "GCN (Graph Convolutional)", **metrics_gcn},
        {"Model": "GAT (Graph Attention)", **metrics_gat},
    ])
    print("\n" + "=" * 75)
    print("BẢNG SO SÁNH HIỆU NĂNG CÁC MÔ HÌNH TRÊN TẬP VALIDATION (400 SINH VIÊN)")
    print("=" * 75)
    print(results_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("=" * 75)

    # Lưu bảng kết quả ra CSV để dùng viết báo cáo
    results_df.to_csv(os.path.join(MODEL_DIR, "experiment_results.csv"), index=False)

    # Lưu trọng số mô hình GAT + thông tin cấu hình đồ thị vào gnn_model.pt
    checkpoint = {
        "model_type": "GAT",
        "in_channels": in_channels,
        "hidden_channels": 16,
        "heads": 4,
        "num_questions": num_q,
        "num_concepts": num_c,
        "state_dict": gat_model.state_dict(),
        "text_features": text_features,
        "edge_index": edge_index
    }
    save_path = os.path.join(MODEL_DIR, "gnn_model.pt")
    torch.save(checkpoint, save_path)
    print(f"\nĐã lưu trọng số mô hình GAT tốt nhất tại: {save_path}")


if __name__ == "__main__":
    main()