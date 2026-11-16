CẤU TRÚC PROJECT:
# project_dacn/
├── share_data/                  # DỮ LIỆU CHUNG (Điểm chốt để cả 3 cùng đọc)
│   ├── concepts.csv              # Đưa 3 file CSV vừa tạo vào đây ngay từ đầu
│   ├── questions.csv
│   └── edges.csv
│
├── data_pipeline/                # LUỒNG 1: Hoàng Hùng Cường (Data & Backend DB)
│   ├── extract_slides.py         # Script đọc slide/giáo trình
│   ├── init_db.py                # Script tạo database.db từ CSV
│   └── database.db               # File SQLite chính thức
│
├── model_gnn/                    # LUỒNG 2: Trần Phạm Phương Trinh (AI & Graph)
│   ├── generate_interactions.py  # Script sinh 1.000 lượt làm bài ảo
│   ├── train_gnn.ipynb           # Notebook Colab huấn luyện Baseline, GCN, GAT
│   ├── inference.py              # Hàm predict_mastery() để Trầm gọi ở Tuần 9
│   └── gnn_model.pt              # Trọng số mô hình sau khi train xong
│
├── web_app/                      # LUỒNG 3: Trần Nguyễn Ngọc Trầm (Frontend Streamlit)
│   ├── app.py                    # Giao diện chính Streamlit
│   ├── graph_viz.py              # Code vẽ đồ thị PyVis xanh/vàng/đỏ
│   └── mock_ai.py                # Hàm giả lập điểm 0.0-1.0 dùng tạm trước Tuần 9
│
├── requirements.txt              # Danh sách thư viện chung của cả nhóm
└── README.md                     # Quy ước chạy project

======================================================================================
QUY TẮC CHIA BRANCH (NHÁNH) TRÊN GIT:
- Nhánh main: sẵn cấu trúc thư mục cho cả project, 3 file mồi trong share_data và file requirements.txt (các cái cần tải)
- 3 nhánh riêng biệt:
    + feat/data-db -> Dành riêng cho Hoàng Hùng Cường.   
    + feat/ai-gnn -> Dành riêng cho Trần Phạm Phương Trinh.   
    + feat/web-ui -> Dành riêng cho Trần Nguyễn Ngọc Trầm. 
- Quy định:
    + Mỗi người chỉ git add, commit và push code nằm bên trong thư mục của mình
    + Không ai được tự ý sửa cột trong các file share_data/*.csv khi chưa họp thống nhất lại cho cả nhóm