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
    "tl": {"cause": "Ano ito", "signs": "Sintomas", "do_now": "Gawin ngayon", "prevention": "Para maiwasan",
           "chemical": "Kung mag-i-spray", "organic": "Organic na paraan", "escalate": "Kailan magtatanong sa MAO",
           "sources": "Sources"},
}

STATUS_MESSAGES = {
    "unsure": {
        "en": "**Not sure.** The model is not confident about this photo. Retake it in daylight: one leaf, "
              "filling most of the frame, in focus, with the spots visible. The closest guesses are listed below.",
        "tl": "**Hindi sigurado.** Hindi sure ang AI sa picture na ito. Kumuha ulit sa liwanag ng araw: "
              "isang dahon lang, halos buong picture, malinaw, at kita ang mga batik. Nasa ibaba ang top 3 hula ng AI.",
    },
    "not_leaf": {
        "en": "**No leaf found.** Take a close photo of a single leaf from the affected plant.",
        "tl": "**Walang nakitang dahon.** Kumuha ng malapitang picture ng isang dahon mula sa halamang may sakit.",
    },
    "other_plant": {
        "en": "**Unsupported plant.** Sipat currently recognises tomato, pepper (sili), potato and corn "
              "leaves only. For other crops, ask your Municipal Agriculture Office.",
        "tl": "**Hindi suportadong halaman.** Kamatis, sili, patatas at mais lang ang kilala ngayon ng SIPAT. "
              "Para sa ibang tanim, magtanong sa Municipal Agriculture Office (MAO).",
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
        return {"en": "No guidance on file for this result.", "tl": "Wala pang gabay para sa resultang ito."}
    return {lang: render(class_key, lang) for lang in ("en", "tl")}
