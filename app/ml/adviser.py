"""Ask a follow-up question about a diagnosis - retrieval first, local model second.

Why this shape (learned the hard way, 2026-10-14): the first version handed the
curated guidance to a 3B local model and told it never to give a dose. Asked
"ilang gramo ng mancozeb sa isang litro?", it answered "500 gramo" anyway, and
an English question made it loop "5.000 words". A small model cannot be trusted
with rules like that, so:

1. RETRIEVAL. The common questions - how much / dose, spraying, organic,
   prevention, what is it, signs, harvesting or selling, when to get help - are
   recognised by keywords (English and Tagalog) and answered with the curated
   text from treatment_data.py, word for word. No model involved: it cannot
   invent anything.
2. MODEL, GUARDED. Anything else goes to the local model (Ollama, qwen2.5:3b)
   with the guidance as its only reference. Its answer is then checked: any
   number with a unit (a dose), a loop, or an empty reply is thrown away and
   replaced with a safe answer pointing to the written guidance and the MAO.

Nothing leaves the PC: no photo is sent anywhere, and Ollama runs locally.
"""

import json
import os
import re
import urllib.error
import urllib.request

from app.ml.classes import CLASSES
from app.ml.treatment import get_treatment_guidance
from app.ml.treatment_data import HARVEST_RULE, LABEL_RULE, TREATMENTS

OLLAMA_URL = os.getenv("OLLAMA_API_URL", "http://127.0.0.1:11434")
MODEL = os.getenv("TANIM_ADVISER_MODEL", "qwen2.5:3b-instruct")
TIMEOUT_S = int(os.getenv("TANIM_ADVISER_TIMEOUT", "90"))
MAX_QUESTION = 400

OFFLINE = {
    "en": "The adviser is offline (Ollama is not running on this computer). The written guidance above still applies.",
    "tl": "Hindi available ang tagapayo (hindi tumatakbo ang Ollama sa computer na ito). Gamitin muna ang nakasulat na gabay sa itaas.",
}
DONT_KNOW = {
    "en": "I can't answer that safely from Tanim Guard's guidance. Read the guidance above, and ask your Municipal Agriculture Office (MAO).",
    "tl": "Hindi ko iyan masasagot nang ligtas mula sa gabay ng Tanim Guard. Basahin ang gabay sa itaas at magtanong sa inyong Municipal Agriculture Office (MAO).",
}

# Order matters: the first intent that matches wins. Dose comes first so
# "ilang gramo ng fungicide" is a dose question, not a spray question.
INTENTS = [
    # \b on the short Tagalog words: "ilan" sits inside "kailangan", "ani" inside "pani...".
    ("dose", r"gramo|\bgrams?\b|\bml\b|litro|liter|litre|kutsara|tablespoon|tbsp|\bdos(e|age)\b|"
             r"how much .*(use|mix|put|spray|apply)|how many (grams|ml|spoons)|"
             r"gaano karami|\bilang\b|\bilan\b|ratio|timpla|\bmix"),
    ("harvest", r"harvest|\bani(hin)?\b|mag-ani|\bbenta|ibenta|\bsell|\beat\b|\bkain(in)?\b|"
                r"\bbunga\b|\bfruits?\b|safe to"),
    ("organic", r"organic|organiko|natural|neem|walang kemikal|no chemical"),
    ("chemical", r"spray|gamot|fungicide|pesticide|insecticide|chemical|kemikal|bactericide|miticide|what to use|ano.*gamitin"),
    ("prevention", r"prevent|iwas|iwasan|maiwasan|next time|susunod|pag-iwas"),
    ("escalate", r"help|tulong|expert|technician|\bmao\b|agriculture office|who.*ask|saan.*magtanong"),
    ("signs", r"sign|symptom|palatandaan|sintomas|itsura|look like|paano malalaman"),
    ("cause", r"what is|ano (ito|iyan|yan|ang sanhi)|sanhi|cause|bakit|why|nakakahawa|spread|kalat"),
    ("do_now", r"what (do|should) i do|anong gagawin|ano.*gawin|gagawin|treat|lunas|cure|gamutin"),
]
_INTENT_RE = [(name, re.compile(pattern, re.I)) for name, pattern in INTENTS]

# A number next to a unit in a model answer is an invented dose.
_DOSE_RE = re.compile(r"\d[\d.,]*\s*(g|gr|gram|gramo|grams|ml|mL|l|litro|liter|litre|kg|kilo|%|cc|tbsp|kutsara)\b", re.I)

_LABELS = {
    "en": {"chemical": "If you spray", "organic": "Organic options", "prevention": "Prevention",
           "cause": "What it is", "signs": "Signs", "do_now": "Do now", "escalate": "When to get help"},
    "tl": {"chemical": "Kung mag-i-spray", "organic": "Organikong paraan", "prevention": "Pag-iwas",
           "cause": "Ano ito", "signs": "Mga palatandaan", "do_now": "Gawin ngayon", "escalate": "Kailan hihingi ng tulong"},
}


def intent_of(question: str) -> str | None:
    for name, pattern in _INTENT_RE:
        if pattern.search(question):
            return name
    return None


def _section(data: dict, field: str, lang: str) -> str | None:
    value = (data.get(field) or {}).get(lang) if isinstance(data.get(field), dict) else None
    if value is None:
        return None
    return " ".join(f"{i}. {s}" for i, s in enumerate(value, 1)) if isinstance(value, list) else value


def retrieve(diagnosis: dict, question: str, lang: str) -> str | None:
    """A curated answer for a recognised question, or None."""
    intent = intent_of(question)
    if intent == "dose":
        return LABEL_RULE[lang] + " " + DONT_KNOW[lang].split(". ", 1)[1]
    if intent == "harvest":
        return HARVEST_RULE[lang]
    data = TREATMENTS.get(diagnosis.get("class_key") or "")
    if intent is None or data is None:
        return None
    text = _section(data, intent, lang)
    if not text:  # e.g. no spray exists for a virus
        text = _section(data, "do_now", lang)
    answer = f"{_LABELS[lang].get(intent, '')}: {text}" if intent in _LABELS[lang] else text
    if intent == "chemical" and data.get("chemical"):
        answer += " " + LABEL_RULE[lang]
    return answer


def is_unsafe(answer: str) -> bool:
    """Reject a model answer that invents a dose, loops, or says nothing."""
    if not answer.strip() or _DOSE_RE.search(answer):
        return True
    lines = [l.strip() for l in answer.splitlines() if l.strip()]
    if lines and max(lines.count(l) for l in lines) >= 3:
        return True
    words = answer.lower().split()
    return len(words) > 20 and len(set(words)) / len(words) < 0.35


def build_messages(diagnosis: dict, question: str, lang: str) -> list[dict]:
    key = diagnosis.get("class_key")
    status = diagnosis.get("status") or "ok"
    reference = get_treatment_guidance(key, status)["en"]
    crop = CLASSES.get(key, {}).get("name", diagnosis.get("disease_name", ""))
    language = "Tagalog (Filipino), simple words" if lang == "tl" else "English, simple words"
    system = (
        "You are Tanim Guard's farm adviser for Filipino smallholder farmers. "
        f"The photo was diagnosed as: {crop} (status: {status}, model confidence "
        f"{diagnosis.get('confidence', 0):.0%}).\n"
        "Answer ONLY from the REFERENCE below. If it is not there, say you do not know "
        "and tell them to ask their Municipal Agriculture Office (MAO). "
        "Never give a dose, amount, mixing ratio, brand or schedule. "
        "If the status is 'unsure', say the photo result is uncertain. "
        f"Answer in two or three short sentences, in {language}.\n\n"
        f"REFERENCE:\n{reference}"
    )
    return [{"role": "system", "content": system},
            {"role": "user", "content": question.strip()[:MAX_QUESTION]}]


def ask(diagnosis: dict, question: str, lang: str = "tl", opener=None) -> dict:
    """{'answer', 'offline', 'source': 'guide'|'model'|'safe'}. Never raises."""
    lang = lang if lang in ("en", "tl") else "tl"
    question = (question or "").strip()[:MAX_QUESTION]
    if not question:
        return {"answer": "", "offline": False, "source": "safe"}

    # No leaf / a plant we do not cover: there is no disease to talk about, and
    # the model's answers here were muddled. Repeat the fixed guidance instead.
    if diagnosis.get("status") in ("not_leaf", "other_plant"):
        text = get_treatment_guidance(None, diagnosis["status"])[lang].replace("**", "")
        return {"answer": text, "offline": False, "source": "guide"}

    curated = retrieve(diagnosis, question, lang)
    if curated:
        return {"answer": curated, "offline": False, "source": "guide"}

    body = json.dumps({"model": MODEL, "stream": False,
                       "messages": build_messages(diagnosis, question, lang),
                       "options": {"temperature": 0.1, "num_predict": 160,
                                   "repeat_penalty": 1.2, "num_ctx": 4096}}).encode("utf-8")
    request = urllib.request.Request(f"{OLLAMA_URL}/api/chat", data=body,
                                     headers={"Content-Type": "application/json"})
    try:
        with (opener or urllib.request.urlopen)(request, timeout=TIMEOUT_S) as response:
            data = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, OSError, ValueError):
        return {"answer": OFFLINE[lang], "offline": True, "source": "safe"}
    answer = ((data.get("message") or {}).get("content") or "").strip()
    if is_unsafe(answer):
        return {"answer": DONT_KNOW[lang], "offline": False, "source": "safe"}
    return {"answer": answer, "offline": False, "source": "model"}
