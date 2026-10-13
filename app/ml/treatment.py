"""Turn a diagnosis into the guidance the result page shows (English + Tagalog).

The content lives in treatment_data.py (curated, sourced). This module only
formats it, and covers the outcomes that are not a crop disease at all.
"""

from app.ml.classes import CLASSES
from app.ml.treatment_data import LABEL_RULE, TREATMENTS

_HEADINGS = {
    "en": {"cause": "What it is", "signs": "Signs", "do_now": "Do now", "prevention": "Prevention",
           "chemical": "If you spray", "organic": "Organic options", "escalate": "When to get help",
           "sources": "Check against"},
    "tl": {"cause": "Ano ito", "signs": "Mga palatandaan", "do_now": "Gawin ngayon", "prevention": "Pag-iwas",
           "chemical": "Kung mag-i-spray", "organic": "Organikong paraan", "escalate": "Kailan hihingi ng tulong",
           "sources": "Batayan"},
}

STATUS_MESSAGES = {
    "unsure": {
        "en": "**Not sure.** The model is not confident about this photo. Retake it in daylight: one leaf, "
              "filling most of the frame, in focus, with the spots visible. The closest guesses are listed below.",
        "tl": "**Hindi sigurado.** Hindi kampante ang model sa larawang ito. Kumuha ulit sa liwanag ng araw: "
              "isang dahon, sakop ang halos buong larawan, malinaw, at kita ang mga batik. Nasa ibaba ang pinakamalapit na hula.",
    },
    "not_leaf": {
        "en": "**No leaf found.** Take a close photo of a single leaf from the affected plant.",
        "tl": "**Walang nakitang dahon.** Kumuha ng malapitang larawan ng isang dahon mula sa apektadong halaman.",
    },
    "other_plant": {
        "en": "**Unsupported plant.** Tanim Guard currently recognises tomato, pepper (sili), potato and corn "
              "leaves only. For other crops, ask your Municipal Agriculture Office.",
        "tl": "**Hindi suportadong halaman.** Kamatis, sili, patatas at mais lamang ang kilala ngayon ng Tanim Guard. "
              "Para sa ibang pananim, magtanong sa inyong Municipal Agriculture Office.",
    },
}


def render(class_key: str, lang: str) -> str:
    data = TREATMENTS[class_key]
    h = _HEADINGS[lang]
    name = CLASSES[class_key]["name" if lang == "en" else "name_tl"]
    out = [f"**{name}**", ""]
    for field in ("cause", "signs"):
        out += [f"**{h[field]}:** {data[field][lang]}", ""]
    for field in ("do_now", "prevention"):
        out.append(f"**{h[field]}:**")
        out += [f"{i}. {step}" for i, step in enumerate(data[field][lang], 1)]
        out.append("")
    for field in ("chemical", "organic"):
        if data.get(field):
            out += [f"**{h[field]}:** {data[field][lang]}", ""]
    if data.get("chemical"):
        out += [f"⚠️ {LABEL_RULE[lang]}", ""]
    out += [f"**{h['escalate']}:** {data['escalate'][lang]}", "",
            f"_{h['sources']}: " + "; ".join(data["sources"]) + "_"]
    return "\n".join(out)


def get_treatment_guidance(class_key: str, status: str = "ok") -> dict:
    """{'en': markdown, 'tl': markdown} for a prediction."""
    if status in STATUS_MESSAGES:
        return dict(STATUS_MESSAGES[status])
    if class_key not in TREATMENTS:
        return {"en": "No guidance on file for this result.", "tl": "Walang gabay para sa resultang ito."}
    return {lang: render(class_key, lang) for lang in ("en", "tl")}
