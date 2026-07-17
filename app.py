"""
⚡ Analog Design KB
Interface Streamlit — RAG sur Drive + Obsidian + vision circuits (Gemini)
"""

import os
import streamlit as st
import google.generativeai as genai
from supabase import create_client
from PIL import Image

# ── Config ────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Analog KB", page_icon="⚡", layout="wide")

def _s(k):
    try:    return st.secrets[k]
    except: return os.environ.get(k, "")

genai.configure(api_key=_s("GOOGLE_API_KEY"))
sb = create_client(_s("SUPABASE_URL"), _s("SUPABASE_KEY"))

CATEGORIES = [
    "Toutes", "opamp", "data_converters", "chopper_offset",
    "bandgap_reference", "power_dcdc_charge_pump", "pll_clock",
    "noise_variability", "layout_matching", "reliability_esd_latchup",
    "sensing_isolated", "trimming", "modeling_tools", "conferences_papers",
    "technos_st", "general_analog_courses", "reference_books",
]

SYSTEM = """Tu es un expert en design de circuits intégrés analogiques CMOS (ST Microelectronics context).
Domaines : op-amp, convertisseurs (ADC/DAC/Sigma-Delta), PLL, chopper, LDO/DC-DC,
bruit/variabilité, layout/matching, fiabilité/ESD, Verilog-A.

Règles :
- Quand des extraits de documents sont fournis, cite leur source exacte.
- Quand une image de circuit est fournie, analyse : topologie, nœuds critiques,
  problèmes potentiels, améliorations suggérées.
- Sois précis, technique. Mentionne les compromis. Utilise des équations si pertinent."""


# ── RAG ───────────────────────────────────────────────────────────────────────

def embed_query(text: str) -> list[float]:
    r = genai.embed_content(
        model="models/text-embedding-004",
        content=text,
        task_type="retrieval_query",
    )
    emb = r["embedding"]
    return emb if isinstance(emb[0], float) else emb[0]

def search(query: str, category: str, k: int) -> list[dict]:
    emb    = embed_query(query)
    params = {"query_embedding": emb, "match_count": k}
    if category != "Toutes":
        params["filter_category"] = category
    return sb.rpc("search_chunks", params).execute().data or []

def build_context(results: list[dict]) -> str:
    if not results: return ""
    parts = ["## Extraits de ta base de connaissances\n"]
    for r in results:
        icon = "🗒️" if r.get("source") == "obsidian" else "📄"
        parts.append(
            f"### {icon} [{r['title']}] — {r['category']} — p.{r['page_num']}\n"
            f"{r['content']}\n"
        )
    return "\n".join(parts)


# ── UI ────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("⚡ Analog KB")
    st.divider()
    category  = st.selectbox("Catégorie", CATEGORIES)
    use_rag   = st.toggle("Chercher dans mes docs", value=True)
    n_chunks  = st.slider("Chunks récupérés", 2, 10, 5)
    st.caption("↑ Plus de chunks = plus de contexte mais plus de tokens")
    st.divider()
    img_file  = st.file_uploader("📷 Capture de circuit", type=["png","jpg","jpeg"])
    if img_file:
        st.image(img_file, use_column_width=True)
    st.divider()
    if st.button("🗑 Nouvelle conversation"):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

if prompt := st.chat_input("Pose ta question…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Recherche RAG
    sources, context = [], ""
    if use_rag:
        with st.spinner("Recherche dans tes docs…"):
            sources = search(prompt, category, n_chunks)
            context = build_context(sources)

    # Prompt complet
    full_prompt = SYSTEM + "\n\n"
    if context:
        full_prompt += context + "\n\n---\n\n"
    full_prompt += f"Question : {prompt}"

    # Appel Gemini (avec ou sans image)
    model    = genai.GenerativeModel("gemini-1.5-flash")
    contents = [full_prompt]
    if img_file:
        img_file.seek(0)
        contents.append(Image.open(img_file))

    with st.chat_message("assistant"):
        with st.spinner("Réflexion…"):
            answer = model.generate_content(contents).text
        st.markdown(answer)

        if sources:
            with st.expander(f"📚 {len(sources)} sources utilisées"):
                for s in sources:
                    icon = "🗒️" if s.get("source") == "obsidian" else "📄"
                    st.markdown(
                        f"{icon} **{s['title']}** — `{s['category']}` — p.{s['page_num']}  \n"
                        f"> {s['content'][:250]}…"
                    )

    st.session_state.messages.append({"role": "assistant", "content": answer})
