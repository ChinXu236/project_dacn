import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv


class WeightedAverageBaseline(nn.Module):
    """
    Mô hình 1: Baseline đối chứng (Tính trung bình cộng có trọng số truyền thống)
    Chỉ tổng hợp điểm từ các câu hỏi nối trực tiếp với Concept (bỏ qua câu chưa làm = 0.5).
    """
    def __init__(self, num_questions, num_concepts, test_edges_df):
        super().__init__()
        W = torch.zeros((num_questions, num_concepts), dtype=torch.float32)
        for _, row in test_edges_df.iterrows():
            q_idx = int(row["source_id"][1:]) - 1  # Q01 -> 0
            c_idx = int(row["target_id"][1:]) - 1  # C01 -> 0
            W[q_idx, c_idx] = float(row["weight"])
        self.register_buffer("W", W)

    def forward(self, interactions):
        answered_mask = (interactions != 0.5).float()
        numerator = torch.matmul(interactions * answered_mask, self.W)
        denominator = torch.matmul(answered_mask, self.W) + 1e-8

        has_answered = (torch.matmul(answered_mask, self.W) > 0).float()
        pred = (numerator / denominator) * has_answered + 0.5 * (1.0 - has_answered)
        return pred


class KnowledgeGCN(nn.Module):
    """
    Mô hình 2: Graph Convolutional Network (GCN)
    """
    def __init__(self, in_channels, hidden_channels=32, num_questions=30, num_concepts=27):
        super().__init__()
        self.num_questions = num_questions
        self.num_concepts = num_concepts
        self.total_nodes = num_questions + num_concepts

        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.head = nn.Sequential(
            nn.Linear(hidden_channels, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x, edge_index, edge_weight=None):
        h = F.relu(self.conv1(x, edge_index, edge_weight=edge_weight))
        h = F.dropout(h, p=0.2, training=self.training)
        h = F.relu(self.conv2(h, edge_index, edge_weight=edge_weight))

        # Tách đúng từng đồ thị trong batch: (batch_size, total_nodes, hidden_channels)
        batch_size = h.size(0) // self.total_nodes
        h_batched = h.view(batch_size, self.total_nodes, -1)

        # Lấy đúng phần node Concept của TẤT CẢ sinh viên trong batch
        concept_h = h_batched[:, self.num_questions : self.total_nodes, :]
        mastery_pred = self.head(concept_h).squeeze(-1)  # (batch_size, num_concepts)
        return mastery_pred


class KnowledgeGAT(nn.Module):
    """
    Mô hình 3: Graph Attention Network (GAT - Mô hình đề xuất chính)
    """
    def __init__(self, in_channels, hidden_channels=16, heads=4, num_questions=30, num_concepts=27):
        super().__init__()
        self.num_questions = num_questions
        self.num_concepts = num_concepts
        self.total_nodes = num_questions + num_concepts

        self.gat1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=0.2)
        self.gat2 = GATConv(hidden_channels * heads, hidden_channels, heads=1, concat=False, dropout=0.2)
        self.head = nn.Sequential(
            nn.Linear(hidden_channels, 16),
            nn.ELU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x, edge_index, return_attention=False):
        if return_attention:
            h, (att_edge_index, att_weights) = self.gat1(x, edge_index, return_attention_weights=True)
        else:
            h = self.gat1(x, edge_index)

        h = F.elu(h)
        h = F.dropout(h, p=0.2, training=self.training)
        h = F.elu(self.gat2(h, edge_index))

        # Tách đúng từng đồ thị trong batch: (batch_size, total_nodes, hidden_channels)
        batch_size = h.size(0) // self.total_nodes
        h_batched = h.view(batch_size, self.total_nodes, -1)

        # Lấy đúng phần node Concept của TẤT CẢ sinh viên trong batch
        concept_h = h_batched[:, self.num_questions : self.total_nodes, :]
        mastery_pred = self.head(concept_h).squeeze(-1)  # (batch_size, num_concepts)

        if return_attention:
            return mastery_pred, (att_edge_index, att_weights)
        return mastery_pred