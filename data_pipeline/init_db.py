import sqlite3
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SHARE = ROOT / "share_data"
DB = Path(__file__).resolve().parent / "database.db"

SCHEMA = """
CREATE TABLE users(
  user_id INTEGER PRIMARY KEY AUTOINCREMENT,
  username TEXT UNIQUE NOT NULL, password_hash TEXT NOT NULL,
  created_at TEXT DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE concepts(
  concept_id TEXT PRIMARY KEY, concept_name TEXT, chapter TEXT, description TEXT,
  slide_file TEXT, slide_pages TEXT, textbook_file TEXT, textbook_pages TEXT);
CREATE TABLE questions(
  question_id TEXT PRIMARY KEY, question_text TEXT,
  option_a TEXT, option_b TEXT, option_c TEXT, option_d TEXT,
  correct_answer TEXT, difficulty TEXT);
CREATE TABLE question_concept_map(
  question_id TEXT, concept_id TEXT, weight REAL,
  PRIMARY KEY(question_id, concept_id));
CREATE TABLE concept_prerequisites(
  source_id TEXT, target_id TEXT, PRIMARY KEY(source_id, target_id));
CREATE TABLE test_sessions(
  session_id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER,
  started_at TEXT DEFAULT CURRENT_TIMESTAMP, submitted_at TEXT, score REAL);
CREATE TABLE user_answers(
  id INTEGER PRIMARY KEY AUTOINCREMENT, session_id INTEGER,
  question_id TEXT, chosen TEXT, is_correct INTEGER);
CREATE TABLE mastery_scores(
  id INTEGER PRIMARY KEY AUTOINCREMENT, session_id INTEGER,
  concept_id TEXT, score REAL);
"""

def has_cycle(pairs):
    out, indeg = {}, {}
    for s, t in pairs:
        out.setdefault(s, []).append(t)
        indeg[t] = indeg.get(t, 0) + 1
        indeg.setdefault(s, 0)
    stack = [n for n, d in indeg.items() if d == 0]
    seen = 0
    while stack:
        n = stack.pop(); seen += 1
        for m in out.get(n, []):
            indeg[m] -= 1
            if indeg[m] == 0:
                stack.append(m)
    return seen != len(indeg)

def main():
    concepts = pd.read_csv(SHARE / "concepts.csv")
    questions = pd.read_csv(SHARE / "questions.csv")
    edges = pd.read_csv(SHARE / "edges.csv")

    test = edges[edges.relation == "tests"].copy()
    pre = edges[edges.relation == "prerequisite"]

    # --- kiểm tra ---
    if has_cycle(list(zip(pre.source_id, pre.target_id))):
        raise SystemExit("LỖI: prerequisite có chu trình!")
    sums = test.groupby("source_id")["weight"].sum()
    bad = sums[(sums - 1).abs() > 1e-6]
    if len(bad):
        print(f"CẢNH BÁO: {len(bad)} câu hỏi có tổng weight != 1.0 -> tự chuẩn hóa trong DB (CSV giữ nguyên):")
        print(bad.to_string())
    test["weight"] = test["weight"] / test.groupby("source_id")["weight"].transform("sum")

    cnt = test.groupby("target_id").size()
    few = [c for c in concepts.concept_id if cnt.get(c, 0) < 4]
    print(f"{len(few)}/{len(concepts)} concept có <4 câu hỏi")
    print("correct_answer phân bố:", questions.correct_answer.value_counts().to_dict())

    if DB.exists():
        DB.unlink()
    con = sqlite3.connect(DB)
    con.executescript(SCHEMA)
    concepts.to_sql("concepts", con, if_exists="append", index=False)
    questions.to_sql("questions", con, if_exists="append", index=False)
    test.rename(columns={"source_id": "question_id", "target_id": "concept_id"})[
        ["question_id", "concept_id", "weight"]].to_sql(
        "question_concept_map", con, if_exists="append", index=False)
    pre[["source_id", "target_id"]].to_sql(
        "concept_prerequisites", con, if_exists="append", index=False)
    con.commit(); con.close()
    print(f"OK -> {DB}")

if __name__ == "__main__":
    main()