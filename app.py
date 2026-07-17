"""
⚡ Analog Design KB
Interface Streamlit — RAG sur Drive + Obsidian + vision circuits (Gemini)
"""

import os
import streamlit as st
from google import genai
from google.genai import types
from supabase import create_client
from PIL import Image

# ── Config ────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Analog KB", page_icon="⚡", layout="wide")

def _s(k):
    try:    return st.secrets[k]
    except: return os.environ.get(k, "")

client = genai.Client(api_key=_s("GOOGLE_API_KEY"))
sb     = create_client(_s("SUPABASE_URL"), _s("SUPABASE_KEY"))

CATEGORIES = [
    "Toutes", "opamp", "data_converters", "chopper_offset",
    "bandgap_reference", "power_dcdc_charge_pump", "pll_clock",
    "noise_variability", "layout_matching", "reliability_esd_latchup",
    "sensing_isolated", "trimming", "modeling_tools", "conferences_papers",
    "technos_st", "general_analog_courses", "reference_books",
]

SYSTEM = """Tu es un expert en design de circuits intégrés analogiques CMOS (contexte ST Microelectronics).
Domaines : op-amp, convertisseurs (ADC/DAC/Sigma-Delta), PLL, chopper, LDO/DC-DC,
bruit/variabilité, layout/matching, fiabilité/ESD, Verilog-A.

Règles :
- Quand des extraits de documents sont fournis, cite leur source exacte (titre + page).
- Quand une image de circuit est fournie, identifie la topologie, les blocs clés,
  les nœuds critiques, les problèmes potentiels.
- Si des documents pertinents ont été trouvés, explique concrètement ce qu'ils apportent
  par rapport à l'architecture présentée.
- Sois précis, technique. Mentionne les compromis. Utilise des équations si pertinent."""

VISION_PROBE = """Tu es un expert en circuits intégrés analogiques.
Analyse cette image de circuit/architecture et génère une description technique dense
(150 mots max) en anglais, listant : topologie, blocs fonctionnels, type de signal,
technologie probable, mots-clés de recherche. Pas de phrase d'intro, juste les faits."""


# ── RAG ───────────────────────────────────────────────────────────────────────

def embed_query(text: str) -> list[float]:
    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=text,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY", output_dimensionality=768),
    )
    return result.embeddings[0].values

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

if prompt := st.chat_input("Pose ta question… ou uploade un circuit et demande une analyse"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    img = None
    circuit_description = None
    if img_file:
        img_file.seek(0)
        img = Image.open(img_file)

    # ── Si image présente : décrire le circuit pour enrichir la requête RAG ──
    if img and use_rag:
        with st.spinner("Analyse du circuit…"):
            probe_resp = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=[VISION_PROBE, img],
            )
            circuit_description = probe_resp.text

    # ── RAG : recherche sur description circuit + question utilisateur ────────
    rag_query = " ".join(filter(None, [circuit_description, prompt]))
    sources, context = [], ""
    if use_rag:
        with st.spinner("Recherche dans tes docs…"):
            sources = search(rag_query, category, n_chunks)
            context = build_context(sources)

    # ── Prompt final ──────────────────────────────────────────────────────────
    full_prompt = SYSTEM + "\n\n"
    if circuit_description:
        full_prompt += f"## Description automatique du circuit uploadé\n{circuit_description}\n\n"
    if context:
        full_prompt += context + "\n\n---\n\n"
    full_prompt += f"Question de l'utilisateur : {prompt}"

    contents: list = [full_prompt]
    if img:
        contents.append(img)

    with st.chat_message("assistant"):
        with st.spinner("Réflexion…"):
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=contents,
            )
            answer = response.text
        st.markdown(answer)

        if sources:
            with st.expander(f"📚 {len(sources)} sources utilisées"):
                for s in sources:
                    icon = "🗒️" if s.get("source") == "obsidian" else "📄"
                    st.markdown(
                        f"{icon} **{s['title']}** — `{s['category']}` — p.{s['page_num']}  \n"
                        f"> {s['content'][:250]}…"
                    )
        if circuit_description:
            with st.expander("🔍 Description circuit utilisée pour la recherche"):
                st.markdown(circuit_description)

    st.session_state.messages.append({"role": "assistant", "content": answer})
