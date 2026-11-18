import hashlib
import json
import re
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "database.db"
DOC_INDEX_PATH = BASE_DIR / "doc_index.json"


def get_connection() -> sqlite3.Connection:
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    return con


# ==============================================================================
# 1. QUẢN LÝ NGƯỜI DÙNG & PHIÊN LÀM BÀI (USERS & TEST SESSIONS)
# ==============================================================================

def _hash_password(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def register_user(username: str, password: str) -> int:
    with get_connection() as con:
        cur = con.execute(
            "INSERT INTO users(username, password_hash) VALUES (?, ?)",
            (username.strip(), _hash_password(password)),
        )
        return cur.lastrowid


def authenticate_user(username: str, password: str) -> Optional[int]:
    with get_connection() as con:
        row = con.execute(
            "SELECT user_id, password_hash FROM users WHERE username = ?",
            (username.strip(),),
        ).fetchone()
        if row and row["password_hash"] == _hash_password(password):
            return row["user_id"]
        return None


def start_test_session(user_id: Optional[int] = None) -> int:
    with get_connection() as con:
        cur = con.execute(
            "INSERT INTO test_sessions(user_id) VALUES (?)",
            (user_id,),
        )
        return cur.lastrowid


# ==============================================================================
# 2. LỌC CÂU HỎI ÔN TẬP & LẤY ĐỀ THI (FILTER QUESTIONS)
# ==============================================================================

def get_all_concepts() -> List[Dict]:
    with get_connection() as con:
        rows = con.execute("SELECT * FROM concepts ORDER BY concept_id").fetchall()
        return [dict(r) for r in rows]


def get_diagnostic_test(limit_per_concept: int = 1, include_answer: bool = False) -> List[Dict]:
    """Lấy bộ câu hỏi chẩn đoán đầu vào phủ đều 27 khái niệm."""
    cols = "q.question_id, q.question_text, q.option_a, q.option_b, q.option_c, q.option_d, q.difficulty"
    if include_answer:
        cols += ", q.correct_answer"

    selected = {}
    with get_connection() as con:
        concepts = [r["concept_id"] for r in con.execute("SELECT concept_id FROM concepts ORDER BY concept_id")]
        for cid in concepts:
            rows = con.execute(
                f"""
                SELECT {cols}
                FROM questions q
                JOIN question_concept_map m ON q.question_id = m.question_id
                WHERE m.concept_id = ?
                ORDER BY m.weight DESC, RANDOM()
                LIMIT ?
                """,
                (cid, limit_per_concept),
            ).fetchall()
            for r in rows:
                selected[r["question_id"]] = dict(r)
    return list(selected.values())


def get_review_questions(
    concept_ids: Optional[List[str]] = None,
    difficulty: Optional[str] = None,
    limit: int = 20,
    include_answer: bool = False,
) -> List[Dict]:
    """Lọc câu hỏi ôn tập theo danh sách concept yếu (từ GNN) và độ khó."""
    cols = "DISTINCT q.question_id, q.question_text, q.option_a, q.option_b, q.option_c, q.option_d, q.difficulty"
    if include_answer:
        cols += ", q.correct_answer"

    query = f"""
        SELECT {cols}
        FROM questions q
        LEFT JOIN question_concept_map m ON q.question_id = m.question_id
        WHERE 1=1
    """
    params = []

    if concept_ids:
        placeholders = ",".join(["?"] * len(concept_ids))
        query += f" AND m.concept_id IN ({placeholders})"
        params.extend(concept_ids)

    if difficulty:
        query += " AND q.difficulty = ?"
        params.append(difficulty)

    query += " ORDER BY RANDOM() LIMIT ?"
    params.append(limit)

    with get_connection() as con:
        rows = con.execute(query, params).fetchall()
        return [dict(r) for r in rows]


# ==============================================================================
# 3. XỬ LÝ LOGIC CHẤM ĐIỂM & TÍNH ĐIỂM THÀNH THẠO SƠ BỘ (GRADING)
# ==============================================================================

def grade_submission(
    answers: Dict[str, str],
    session_id: Optional[int] = None,
) -> Dict:
    """
    Chấm điểm bài làm từ dict {question_id: chosen_option ('A'|'B'|'C'|'D')}.
    - Lưu kết quả vào user_answers, test_sessions và mastery_scores (nếu có session_id).
    - Trả về tổng điểm (thang 10), chi tiết từng câu và điểm thành thạo sơ bộ theo từng concept.
    """
    if not answers:
        return {"total_questions": 0, "correct_count": 0, "score_10": 0.0, "details": [], "concept_scores": {}}

    q_ids = list(answers.keys())
    placeholders = ",".join(["?"] * len(q_ids))

    with get_connection() as con:
        q_rows = con.execute(
            f"SELECT question_id, correct_answer, difficulty FROM questions WHERE question_id IN ({placeholders})",
            q_ids,
        ).fetchall()
        correct_map = {r["question_id"]: r["correct_answer"] for r in q_rows}

        map_rows = con.execute(
            f"SELECT question_id, concept_id, weight FROM question_concept_map WHERE question_id IN ({placeholders})",
            q_ids,
        ).fetchall()

        details = []
        correct_count = 0
        for qid, chosen in answers.items():
            chosen_clean = str(chosen).strip().upper()
            actual = correct_map.get(qid, "")
            is_correct = 1 if (actual and chosen_clean == actual) else 0
            correct_count += is_correct
            details.append({
                "question_id": qid,
                "chosen": chosen_clean,
                "correct_answer": actual,
                "is_correct": bool(is_correct),
            })

        total = len(details)
        score_10 = round((correct_count / total) * 10.0, 2) if total > 0 else 0.0

        # Tính điểm thành thạo sơ bộ (0.0 - 1.0) theo trọng số weight của từng concept
        concept_weighted_earned = {}
        concept_weighted_total = {}
        detail_lookup = {d["question_id"]: (1.0 if d["is_correct"] else 0.0) for d in details}

        for r in map_rows:
            qid = r["question_id"]
            cid = r["concept_id"]
            w = float(r["weight"])
            if qid in detail_lookup:
                concept_weighted_earned[cid] = concept_weighted_earned.get(cid, 0.0) + w * detail_lookup[qid]
                concept_weighted_total[cid] = concept_weighted_total.get(cid, 0.0) + w

        concept_scores = {
            cid: round(concept_weighted_earned[cid] / concept_weighted_total[cid], 4)
            for cid in sorted(concept_weighted_total.keys())
            if concept_weighted_total[cid] > 0
        }

        # Lưu vào database nếu truyền session_id
        if session_id is not None:
            con.execute(
                "UPDATE test_sessions SET submitted_at = CURRENT_TIMESTAMP, score = ? WHERE session_id = ?",
                (score_10, session_id),
            )
            con.executemany(
                "INSERT INTO user_answers(session_id, question_id, chosen, is_correct) VALUES (?, ?, ?, ?)",
                [(session_id, d["question_id"], d["chosen"], int(d["is_correct"])) for d in details],
            )
            con.executemany(
                "INSERT INTO mastery_scores(session_id, concept_id, score) VALUES (?, ?, ?)",
                [(session_id, cid, sc) for cid, sc in concept_scores.items()],
            )

    return {
        "session_id": session_id,
        "total_questions": total,
        "correct_count": correct_count,
        "score_10": score_10,
        "details": details,
        "concept_scores": concept_scores,
    }


def save_gnn_mastery_scores(session_id: int, gnn_scores: Dict[str, float]) -> None:
    """Lưu hoặc cập nhật điểm thành thạo sau khi mô hình GNN suy luận."""
    with get_connection() as con:
        con.execute("DELETE FROM mastery_scores WHERE session_id = ?", (session_id,))
        con.executemany(
            "INSERT INTO mastery_scores(session_id, concept_id, score) VALUES (?, ?, ?)",
            [(session_id, cid, float(sc)) for cid, sc in gnn_scores.items()],
        )


# ==============================================================================
# 4. TRÍCH XUẤT NGUỒN TÀI LIỆU THAM KHẢO (SLIDE & GIÁO TRÌNH TỪ DOC_INDEX.JSON)
# ==============================================================================

def _parse_page_numbers(page_str: str) -> List[int]:
    cleaned = re.sub(r"\(.*?\)", "", str(page_str))
    pages = []
    for part in cleaned.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            if a.strip().isdigit() and b.strip().isdigit():
                pages.extend(range(int(a), int(b) + 1))
        elif part.isdigit():
            pages.append(int(part))
    return sorted(set(pages))


def _load_doc_index() -> Dict[tuple, str]:
    if not DOC_INDEX_PATH.exists():
        return {}
    raw = json.loads(DOC_INDEX_PATH.read_text(encoding="utf-8"))
    return {(item["file"], int(item["page"])): item["text"] for item in raw}


def get_concept_references(concept_id: str, max_excerpt_chars: int = 600) -> Optional[Dict]:
    """
    Trích xuất thông tin slide, giáo trình và nội dung văn bản tương ứng từ doc_index.json
    cho một concept_id cụ thể.
    """
    with get_connection() as con:
        row = con.execute("SELECT * FROM concepts WHERE concept_id = ?", (concept_id,)).fetchone()
        if not row:
            return None
        concept = dict(row)

        prereqs = [
            dict(r)
            for r in con.execute(
                """
                SELECT c.concept_id, c.concept_name
                FROM concept_prerequisites p
                JOIN concepts c ON p.source_id = c.concept_id
                WHERE p.target_id = ?
                """,
                (concept_id,),
            ).fetchall()
        ]

    doc_map = _load_doc_index()

    slide_pages = _parse_page_numbers(concept["slide_pages"])
    slide_excerpts = [
        {"page": p, "text": doc_map.get((concept["slide_file"], p), "")[:max_excerpt_chars]}
        for p in slide_pages
        if (concept["slide_file"], p) in doc_map
    ]

    textbook_pages = _parse_page_numbers(concept["textbook_pages"])
    textbook_excerpts = [
        {"page": p, "text": doc_map.get((concept["textbook_file"], p), "")[:max_excerpt_chars]}
        for p in textbook_pages
        if (concept["textbook_file"], p) in doc_map
    ]

    return {
        "concept_id": concept["concept_id"],
        "concept_name": concept["concept_name"],
        "chapter": concept["chapter"],
        "description": concept["description"],
        "prerequisites": prereqs,
        "slide": {
            "file": concept["slide_file"],
            "pages_label": concept["slide_pages"],
            "excerpts": slide_excerpts,
        },
        "textbook": {
            "file": concept["textbook_file"],
            "pages_label": concept["textbook_pages"],
            "excerpts": textbook_excerpts,
        },
    }


def get_question_references(question_id: str) -> List[Dict]:
    """Lấy toàn bộ nguồn tài liệu tham khảo của các khái niệm gắn với một câu hỏi."""
    with get_connection() as con:
        rows = con.execute(
            "SELECT concept_id FROM question_concept_map WHERE question_id = ? ORDER BY weight DESC",
            (question_id,),
        ).fetchall()
    return [ref for r in rows if (ref := get_concept_references(r["concept_id"])) is not None]


# ==============================================================================
# KIỂM THỬ NHANH KHI CHẠY TRỰC TIẾP FILE
# ==============================================================================
if __name__ == "__main__":
    print("1. Test lọc câu hỏi ôn tập cho C08 (Đường găng CPM):")
    qs = get_review_questions(concept_ids=["C08"], limit=3, include_answer=True)
    for q in qs:
        print(f"   - {q['question_id']} [{q['difficulty']}]: {q['question_text'][:60]}... -> Đáp án: {q['correct_answer']}")

    print("\n2. Test chấm điểm giả lập:")
    sample_answers = {q["question_id"]: q["correct_answer"] for q in qs}
    res = grade_submission(sample_answers)
    print(f"   - Điểm: {res['score_10']}/10 | Concept scores: {res['concept_scores']}")

    print("\n3. Test trích xuất nguồn tài liệu cho C08:")
    ref = get_concept_references("C08")
    print(f"   - {ref['concept_name']} | Slide: {ref['slide']['file']} (tr.{ref['slide']['pages_label']}) | Giáo trình: {ref['textbook']['file']} (tr.{ref['textbook']['pages_label']})")
    if ref["slide"]["excerpts"]:
        print(f"   - Trích đoạn slide trang {ref['slide']['excerpts'][0]['page']}: {ref['slide']['excerpts'][0]['text'][:120]}...")