"""
⚡ Analog Design KB
Interface Streamlit — RAG sur Drive + Obsidian + vision circuits (Gemini)
"""

import os, json
import streamlit as st
from google import genai
from google.genai import types
from supabase import create_client
from PIL import Image

try:
    from streamlit_paste_button import paste_image_button
    HAS_PASTE = True
except ImportError:
    HAS_PASTE = False

# ── Config ────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Analog KB", page_icon="⚡", layout="wide")

def _s(k):
    try:    return st.secrets[k]
    except: return os.environ.get(k, "")

client = genai.Client(api_key=_s("GOOGLE_API_KEY"))
sb     = create_client(_s("SUPABASE_URL"), _s("SUPABASE_KEY"))

ALL_CATEGORIES = [
    "opamp", "data_converters", "chopper_offset",
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

def search(query: str, categories: list[str], k: int) -> list[dict]:
    emb = embed_query(query)
    if not categories:
        return sb.rpc("search_chunks", {"query_embedding": emb, "match_count": k}).execute().data or []
    # Multi-catégorie : une requête par catégorie, fusion + dédup
    seen, results = set(), []
    per_cat = max(2, k // len(categories))
    for cat in categories:
        for r in (sb.rpc("search_chunks", {"query_embedding": emb, "match_count": per_cat, "filter_category": cat}).execute().data or []):
            if r["id"] not in seen:
                results.append(r)
                seen.add(r["id"])
    return sorted(results, key=lambda x: x.get("similarity", 0), reverse=True)[:k]

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

def suggest_categories(project_desc: str) -> list[str]:
    prompt = f"""Catégories disponibles : {', '.join(ALL_CATEGORIES)}

Projet : "{project_desc}"

Retourne UNIQUEMENT un tableau JSON des 2-4 catégories les plus pertinentes parmi la liste. Ex: ["data_converters","noise_variability"]"""
    r = client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
    try:
        text = r.text.strip()
        if "```" in text:
            text = text.split("```")[1].replace("json","").strip()
        return [c for c in json.loads(text) if c in ALL_CATEGORIES]
    except:
        return []


# ── State ─────────────────────────────────────────────────────────────────────

for k, v in [("messages",[]), ("active_categories",[]), ("pasted_img",None)]:
    if k not in st.session_state:
        st.session_state[k] = v


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("⚡ Analog KB")
    st.divider()

    # Contexte projet → suggestion catégories
    st.markdown("**Sur quoi tu travailles ?**")
    project_ctx = st.text_input("", placeholder="ex: sigma-delta ADC automotive",
                                label_visibility="collapsed")
    if st.button("🎯 Suggérer catégories", use_container_width=True) and project_ctx:
        with st.spinner("Analyse…"):
            st.session_state.active_categories = suggest_categories(project_ctx)

    active_cats = st.multiselect(
        "Catégories actives (vide = toutes)",
        ALL_CATEGORIES,
        default=st.session_state.active_categories,
    )
    st.session_state.active_categories = active_cats

    use_rag  = st.toggle("Chercher dans mes docs", value=True)
    n_chunks = st.slider("Chunks récupérés", 2, 10, 5)
    st.caption("↑ Plus de chunks = plus de contexte mais plus de tokens")
    st.divider()

    # Image : upload ou coller
    st.markdown("**📷 Circuit**")
    img_file = st.file_uploader("Upload", type=["png","jpg","jpeg"],
                                label_visibility="collapsed")
    if HAS_PASTE:
        paste_result = paste_image_button("📋 Coller (Ctrl+V)", key="paste_btn")
        if paste_result.image_data is not None:
            st.session_state.pasted_img = paste_result.image_data

    active_img = None
    if img_file:
        img_file.seek(0)
        active_img = Image.open(img_file)
    elif st.session_state.pasted_img:
        active_img = st.session_state.pasted_img

    if active_img:
        st.image(active_img, use_column_width=True)
        if st.button("✕ Effacer l'image"):
            st.session_state.pasted_img = None
            st.rerun()

    st.divider()
    if st.button("🗑 Nouvelle conversation"):
        st.session_state.messages = []
        st.rerun()


# ── Chat ──────────────────────────────────────────────────────────────────────

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

if prompt := st.chat_input("Pose ta question… ou uploade/colle un circuit"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    img = active_img
    circuit_description = None

    if img and use_rag:
        with st.spinner("Analyse du circuit…"):
            circuit_description = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=[VISION_PROBE, img],
            ).text

    rag_query = " ".join(filter(None, [circuit_description, prompt]))
    sources, context = [], ""
    if use_rag:
        with st.spinner("Recherche dans tes docs…"):
            sources = search(rag_query, st.session_state.active_categories, n_chunks)
            context = build_context(sources)

    full_prompt = SYSTEM + "\n\n"
    if circuit_description:
        full_prompt += f"## Description automatique du circuit\n{circuit_description}\n\n"
    if context:
        full_prompt += context + "\n\n---\n\n"
    full_prompt += f"Question : {prompt}"

    contents: list = [full_prompt]
    if img:
        contents.append(img)

    with st.chat_message("assistant"):
        with st.spinner("Réflexion…"):
            answer = client.models.generate_content(
                model="gemini-2.0-flash", contents=contents,
            ).text
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
            with st.expander("🔍 Description circuit (requête RAG)"):
                st.markdown(circuit_description)

    st.session_state.messages.append({"role": "assistant", "content": answer})
