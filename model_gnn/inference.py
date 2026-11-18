import os
import torch
import numpy as np
import pandas as pd
from gnn_model import KnowledgeGAT

MODEL_PATH = os.path.join(os.path.dirname(__file__), "gnn_model.pt")
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "share_data")


class GNNMasteryPredictor:
    """
    Class nạp mô hình GAT đã huấn luyện để suy luận (inference) điểm thành thạo
    cho từng sinh viên và trích xuất trọng số chú ý (Attention Weights).
    """
    def __init__(self, model_path=MODEL_PATH):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

        self.num_q = checkpoint["num_questions"]
        self.num_c = checkpoint["num_concepts"]
        self.text_features = checkpoint["text_features"]
        self.edge_index = checkpoint["edge_index"].to(self.device)

        self.model = KnowledgeGAT(
            in_channels=checkpoint["in_channels"],
            hidden_channels=checkpoint["hidden_channels"],
            heads=checkpoint["heads"],
            num_questions=self.num_q,
            num_concepts=self.num_c
        ).to(self.device)

        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()

        # Chuẩn bị sẵn ma trận node_type cố định
        num_nodes = self.num_q + self.num_c
        self.node_type = np.zeros((num_nodes, 2), dtype=np.float32)
        self.node_type[:self.num_q, 0] = 1.0
        self.node_type[self.num_q:, 1] = 1.0

    def predict(self, student_answers_dict, return_attention=False):
        """
        Đầu vào:
            student_answers_dict: dict dạng {"Q01": 1.0, "Q02": 0.0, ...}
            (Câu nào không có trong dict mặc định là 0.5 - Chưa làm)
        Đầu ra:
            dict dạng {"C01": 0.82, "C02": 0.41, ...}
        """
        interactions = np.full(self.num_q, 0.5, dtype=np.float32)
        for qid, val in student_answers_dict.items():
            q_idx = int(qid[1:]) - 1
            if 0 <= q_idx < self.num_q:
                interactions[q_idx] = float(val)

        num_nodes = self.num_q + self.num_c
        status_feat = np.full((num_nodes, 1), 0.5, dtype=np.float32)
        status_feat[:self.num_q, 0] = interactions

        x_np = np.hstack([status_feat, self.node_type, self.text_features]).astype(np.float32)
        x_tensor = torch.tensor(x_np, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            if return_attention:
                preds, att_info = self.model(x_tensor, self.edge_index, return_attention=True)
            else:
                preds = self.model(x_tensor, self.edge_index)

        scores = preds.squeeze(0).cpu().numpy()
        result = {f"C{i+1:02d}": round(float(scores[i]), 4) for i in range(self.num_c)}

        if return_attention:
            return result, att_info
        return result


# Hàm tiện ích gọi nhanh cho Web Streamlit (Trầm chỉ cần import hàm này)
def predict_mastery(student_answers_dict):
    predictor = GNNMasteryPredictor()
    return predictor.predict(student_answers_dict)


# Chạy kiểm thử nhanh kịch bản sinh viên hổng kiến thức Bài 3 (CPM) & Bài 4 (EVM)
if __name__ == "__main__":
    concepts_df = pd.read_csv(os.path.join(DATA_DIR, "concepts.csv"))
    # Giả lập 1 sinh viên làm đúng Bài 1, Bài 2 nhưng làm sai toàn bộ Bài 3 (Q07, Q08, Q09) và Bài 4 (Q12)
    sample_test = {
        "Q01": 1.0, "Q02": 1.0, "Q04": 1.0, "Q05": 1.0,
        "Q07": 0.0, "Q08": 0.0, "Q09": 0.0, "Q12": 0.0
    }
    mastery_out = predict_mastery(sample_test)

    print("--- KẾT QUẢ SUY LUẬN THỬ NGHIỆM CỦA MÔ HÌNH GAT ---")
    for _, row in concepts_df.iterrows():
        cid = row["concept_id"]
        score = mastery_out[cid]
        status = "ĐỎ (Hổng)" if score < 0.5 else ("VÀNG (Trung bình)" if score < 0.7 else "XANH (Vững)")
        print(f"{cid} | {row['concept_name']:<55} | Điểm: {score:.4f} -> {status}")