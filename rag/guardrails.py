# app/rag/guardrails.py
import re

# ---------- Messages ----------
EMERGENCY_MSG = (
    "⚠️ Bei akuten Notfällen rufen Sie bitte umgehend den Notruf 112. "
    "Für medizinische Beratung erreichen Sie AOK-Clarimedis rund um die Uhr."
)

OUT_OF_SCOPE_MSG = (
    "Dazu liegen mir in den bereitgestellten AOK-Quellen keine Informationen vor. "
    "Bitte stellen Sie eine Frage rund um gesetzliche Krankenversicherung, AOK-Leistungen, ePA, "
    "Beiträge, Mitgliedschaft oder Versichertenservices."
)

MEMBER_SPECIFIC_MSG = (
    "Für personenbezogene Anliegen (z. B. Antrags- oder Leistungsstatus, Adress-/Bankdaten) "
    "nutzen Sie bitte 'Meine AOK' oder rufen Sie die kostenfreie Servicenummer 0800 026 00 00 an."
)

# ---------- Heuristics ----------
_EMERGENCY = re.compile(
    r"\b(notfall|akut|bewusstlos|atemnot|starker(?:\s|-)schmerz|blutung|schlaganfall|herzinfarkt)\b",
    re.I,
)

_OUT_OF_SCOPE = re.compile(
    r"\b(programmieren|python|fußball|reise|kfz|steuererklärung|mathematikprüfung|gaming|wetter)\b",
    re.I,
)

_MEMBER_SPECIFIC = re.compile(
    r"\b(mein(?:e|) antrag|leistungsstand|bearbeitungsstand|mitgliedsnummer|"
    r"adresse ändern|iban|bankverbindung|krankengeld status|"
    r"rechnung einreichen status)\b",
    re.I,
)

def is_emergency(text: str) -> bool:
    return bool(_EMERGENCY.search(text or ""))

def is_out_of_scope(text: str) -> bool:
    return bool(_OUT_OF_SCOPE.search(text or ""))

def is_member_specific(text: str) -> bool:
    return bool(_MEMBER_SPECIFIC.search(text or ""))

def needs_fallback_from_retrieval(num_docs: int, min_needed: int = 1) -> bool:
    """If retrieval returns too few docs, trigger a safe fallback."""
    return num_docs < min_needed