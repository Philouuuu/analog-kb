#!/usr/bin/env python3
"""
analog-kb — pipeline d'ingestion
Lit PDFs (Drive) et Markdown (Obsidian/GitHub), embède avec Gemini,
stocke dans Supabase.

Setup :
  pip install -r requirements-ingest.txt
  cp .env.example .env  # remplis les valeurs
  source .env  (Linux/Mac) ou set les vars (Windows)

Usage :
  # Indexer tous les PDFs du dossier ST Drive :
  python ingest.py --source-dir "G:/Mon Drive/ST" --type pdf

  # Indexer tes notes Obsidian :
  python ingest.py --source-dir "C:/Users/phili/obsidian/analog-kb" --type md

  # Relancer = ne re-traite que les nouveaux fichiers (idempotent)
"""

import os, sys, time, argparse
from pathlib import Path

import pdfplumber
import google.generativeai as genai
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY   = os.environ["GOOGLE_API_KEY"]
SUPABASE_URL     = os.environ["SUPABASE_URL"]
SUPABASE_SERVICE = os.environ["SUPABASE_SERVICE_KEY"]

genai.configure(api_key=GOOGLE_API_KEY)
sb = create_client(SUPABASE_URL, SUPABASE_SERVICE)

CATEGORIES = {
    "reference_books","data_converters","opamp","chopper_offset",
    "bandgap_reference","power_dcdc_charge_pump","pll_clock",
    "noise_variability","layout_matching","reliability_esd_latchup",
    "sensing_isolated","trimming","modeling_tools","conferences_papers",
    "technos_st","general_analog_courses","general",
}

CHUNK_CHARS   = 1200
OVERLAP_CHARS = 150
BATCH_SIZE    = 20
BATCH_DELAY   = 1.2   # sec — respecte le rate limit Gemini free tier


# ── Utilitaires ───────────────────────────────────────────────────────────────

def infer_category(path: Path) -> str:
    for part in path.parts:
        if part in CATEGORIES:
            return part
    return "general"

def chunk_text(text: str, page_num: int) -> list[dict]:
    chunks, start, idx = [], 0, 0
    while start < len(text):
        blob = text[start : start + CHUNK_CHARS].strip()
        if blob:
            chunks.append({"content": blob, "page_num": page_num, "chunk_index": idx})
            idx += 1
        start += CHUNK_CHARS - OVERLAP_CHARS
    return chunks

def embed_texts(texts: list[str], task: str = "retrieval_document") -> list[list[float]]:
    r = genai.embed_content(
        model="models/text-embedding-004",
        content=texts,
        task_type=task,
    )
    embs = r["embedding"]
    # L'API renvoie une liste de listes ou une seule liste selon le batch
    if embs and isinstance(embs[0], float):
        embs = [embs]
    return embs

def already_indexed(file_path: str) -> bool:
    r = sb.table("documents").select("id").eq("file_path", file_path).execute()
    return len(r.data) > 0

def insert_chunks(doc_id: str, chunks: list[dict], embeddings: list[list[float]]):
    rows = [
        {"document_id": doc_id, "content": c["content"],
         "embedding": emb, "page_num": c["page_num"], "chunk_index": c["chunk_index"]}
        for c, emb in zip(chunks, embeddings)
    ]
    for i in range(0, len(rows), 50):
        sb.table("chunks").insert(rows[i : i+50]).execute()

def embed_all(texts: list[str]) -> list[list[float]]:
    all_embs = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        all_embs.extend(embed_texts(batch))
        print(f"    → {min(i+BATCH_SIZE, len(texts))}/{len(texts)} chunks embédés")
        if i + BATCH_SIZE < len(texts):
            time.sleep(BATCH_DELAY)
    return all_embs


# ── Traitement PDF ─────────────────────────────────────────────────────────────

def process_pdf(path: Path, root: Path):
    rel = str(path.relative_to(root))
    if already_indexed(rel):
        print(f"  ⏭  {path.name}")
        return

    print(f"  📄 {path.name}")
    chunks = []
    try:
        with pdfplumber.open(path) as pdf:
            for i, page in enumerate(pdf.pages, 1):
                text = page.extract_text() or ""
                if text.strip():
                    chunks.extend(chunk_text(text, i))
    except Exception as e:
        print(f"    ⚠️  Erreur : {e}")
        return

    if not chunks:
        print("    ⚠️  Aucun texte extrait (PDF scanné ?)")
        return

    doc = sb.table("documents").insert({
        "title": path.stem, "source": "drive",
        "category": infer_category(path), "file_path": rel, "file_type": "pdf",
    }).execute()
    doc_id = doc.data[0]["id"]

    embeddings = embed_all([c["content"] for c in chunks])
    insert_chunks(doc_id, chunks, embeddings)
    print(f"    ✅ {len(chunks)} chunks → {infer_category(path)}")


# ── Traitement Markdown ────────────────────────────────────────────────────────

def process_md(path: Path, root: Path, source: str):
    rel  = str(path.relative_to(root))
    text = path.read_text(encoding="utf-8", errors="ignore").strip()
    if not text or already_indexed(rel):
        return

    print(f"  📝 {path.name}")
    chunks = chunk_text(text, page_num=1)

    doc = sb.table("documents").insert({
        "title": path.stem, "source": source,
        "category": infer_category(path), "file_path": rel, "file_type": "md",
    }).execute()
    doc_id = doc.data[0]["id"]

    embeddings = embed_all([c["content"] for c in chunks])
    insert_chunks(doc_id, chunks, embeddings)
    print(f"    ✅ {len(chunks)} chunks → {source}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-dir", required=True)
    ap.add_argument("--type",       choices=["pdf","md","all"], default="all")
    ap.add_argument("--category",   help="Limiter à un sous-dossier catégorie")
    args = ap.parse_args()

    root = Path(args.source_dir)
    scan = root / args.category if args.category else root
    source_type = "obsidian" if "obsidian" in str(root).lower() else "github"

    if args.type in ("pdf","all"):
        pdfs = sorted(scan.rglob("*.pdf"))
        print(f"\n📚 {len(pdfs)} PDFs trouvés dans {scan}")
        for p in pdfs:
            process_pdf(p, root)

    if args.type in ("md","all"):
        mds = sorted(scan.rglob("*.md"))
        print(f"\n📓 {len(mds)} Markdown trouvés dans {scan}")
        for m in mds:
            process_md(m, root, source_type)

    print("\n🎉 Ingestion terminée !")

if __name__ == "__main__":
    main()
