"""Single-file RAG over ./data using Azure OpenAI.

Usage:
    pip install openai numpy pypdf
    python main.py                      # interactive Q&A
    python main.py "질문 내용"            # one-shot
    python main.py --rebuild            # force re-embedding

Required env vars (see README / shell exports):
    AOAI_ENDPOINT, AOAI_API_KEY,
    AOAI_DEPLOY_GPT4O_MINI (chat), AOAI_DEPLOY_EMBED_3_SMALL (embedding)
Optional:
    AOAI_API_VERSION (default 2024-10-21)
    RAG_CHAT_DEPLOY / RAG_EMBED_DEPLOY to override deployments
    RAG_DATA_DIR (default ./data), RAG_TOP_K (default 5)
"""

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
from openai import AzureOpenAI

ROOT = Path(__file__).resolve().parent
DATA_DIR = Path(os.getenv("RAG_DATA_DIR", ROOT / "data"))
CACHE_DIR = ROOT / ".rag_cache"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120
TOP_K = int(os.getenv("RAG_TOP_K", "5"))
EMBED_BATCH = 64


def env(name: str, default: str | None = None) -> str:
    value = os.getenv(name, default)
    if not value:
        sys.exit(f"[error] 환경변수 {name} 가 설정되지 않았습니다.")
    return value


CHAT_DEPLOY = os.getenv("RAG_CHAT_DEPLOY") or env("AOAI_DEPLOY_GPT4O_MINI")
EMBED_DEPLOY = os.getenv("RAG_EMBED_DEPLOY") or env("AOAI_DEPLOY_EMBED_3_SMALL")

client = AzureOpenAI(
    azure_endpoint=env("AOAI_ENDPOINT"),
    api_key=env("AOAI_API_KEY"),
    api_version=os.getenv("AOAI_API_VERSION", "2024-10-21"),
)


# ---------- 1. Load ----------
def load_documents(data_dir: Path) -> list[dict]:
    docs = []
    for path in sorted(data_dir.rglob("*")):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix in {".md", ".txt"}:
            text = path.read_text(encoding="utf-8", errors="ignore")
        elif suffix == ".pdf":
            from pypdf import PdfReader

            try:
                reader = PdfReader(str(path))
                text = "\n".join(page.extract_text() or "" for page in reader.pages)
            except Exception as e:  # broken / encrypted PDF
                print(f"[warn] PDF 로드 실패 {path}: {e}")
                continue
        else:
            continue
        if text.strip():
            docs.append({"source": str(path.relative_to(data_dir)), "text": text})
    return docs


# ---------- 2. Chunk ----------
def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    text = " ".join(text.split())
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start : start + size])
        start += size - overlap
    return chunks


# ---------- 3. Embed / Index ----------
def embed(texts: list[str]) -> np.ndarray:
    vectors = []
    for i in range(0, len(texts), EMBED_BATCH):
        resp = client.embeddings.create(model=EMBED_DEPLOY, input=texts[i : i + EMBED_BATCH])
        vectors.extend(d.embedding for d in resp.data)
    arr = np.array(vectors, dtype=np.float32)
    return arr / np.linalg.norm(arr, axis=1, keepdims=True)


def build_index(rebuild: bool = False) -> tuple[list[dict], np.ndarray]:
    docs = load_documents(DATA_DIR)
    if not docs:
        sys.exit(f"[error] {DATA_DIR} 에 읽을 문서(.md/.txt/.pdf)가 없습니다.")

    chunks = [
        {"source": d["source"], "text": c}
        for d in docs
        for c in chunk_text(d["text"])
    ]
    fingerprint = hashlib.sha256(
        (EMBED_DEPLOY + json.dumps(chunks, ensure_ascii=False)).encode()
    ).hexdigest()[:16]
    meta_path = CACHE_DIR / f"{fingerprint}.json"
    vec_path = CACHE_DIR / f"{fingerprint}.npy"

    if not rebuild and meta_path.exists() and vec_path.exists():
        print(f"[index] 캐시 사용: {len(chunks)} chunks")
        return json.loads(meta_path.read_text(encoding="utf-8")), np.load(vec_path)

    print(f"[index] {len(docs)} docs → {len(chunks)} chunks 임베딩 중...")
    vectors = embed([c["text"] for c in chunks])
    CACHE_DIR.mkdir(exist_ok=True)
    meta_path.write_text(json.dumps(chunks, ensure_ascii=False), encoding="utf-8")
    np.save(vec_path, vectors)
    return chunks, vectors


# ---------- 4. Retrieve ----------
def retrieve(query: str, chunks: list[dict], vectors: np.ndarray, k: int = TOP_K) -> list[dict]:
    q = embed([query])[0]
    scores = vectors @ q
    top = np.argsort(-scores)[:k]
    return [{**chunks[i], "score": float(scores[i])} for i in top]


# ---------- 5. Generate ----------
SYSTEM_PROMPT = (
    "당신은 물류/유통 AI 스타트업 투자 분석가입니다. "
    "반드시 제공된 [컨텍스트]에 근거해서 한국어로 답하세요. "
    "근거가 없으면 '자료에서 확인되지 않습니다'라고 답하고, "
    "답변 끝에 사용한 출처 번호를 [1], [2] 형태로 표시하세요."
)


def answer(query: str, chunks: list[dict], vectors: np.ndarray) -> str:
    hits = retrieve(query, chunks, vectors)
    context = "\n\n".join(
        f"[{i}] (source: {h['source']})\n{h['text']}" for i, h in enumerate(hits, 1)
    )
    resp = client.chat.completions.create(
        model=CHAT_DEPLOY,
        temperature=0.2,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"[컨텍스트]\n{context}\n\n[질문]\n{query}"},
        ],
    )
    sources = "\n".join(
        f"  [{i}] {h['source']} (score={h['score']:.3f})" for i, h in enumerate(hits, 1)
    )
    return f"{resp.choices[0].message.content}\n\n출처:\n{sources}"


def main() -> None:
    args = sys.argv[1:]
    rebuild = "--rebuild" in args
    args = [a for a in args if a != "--rebuild"]

    chunks, vectors = build_index(rebuild=rebuild)

    if args:
        print(answer(" ".join(args), chunks, vectors))
        return

    print("질문을 입력하세요 (종료: exit / Ctrl+D)")
    while True:
        try:
            query = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if query.lower() in {"exit", "quit", "q"}:
            break
        if query:
            print(answer(query, chunks, vectors))


if __name__ == "__main__":
    main()
