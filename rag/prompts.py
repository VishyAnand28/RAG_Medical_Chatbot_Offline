# app/rag/prompts.py

from __future__ import annotations
from textwrap import dedent

SYSTEM_DE = dedent("""
Du bist ein sachlicher Assistent der AOK.
Antworte NUR auf Basis der bereitgestellten Kontexte. Wenn die Kontexte nicht reichen,
sage: "Dafür habe ich in den bereitgestellten Quellen keinen Beleg."
Regeln:
- Keine Diagnosen oder individuelle Therapieempfehlungen.
- Keine URLs erfinden; zitiere nur die bereitgestellten Quellen (die UI fügt sie an).
- Antworte kurz und klar auf Deutsch (3–6 Sätze, ggf. 1–2 Aufzählungspunkte).
- Wenn es um persönliche Anliegen (Mitgliedsdaten, Anträge, Leistungsstände) geht:
  verweise auf "Meine AOK" oder die Servicenummer 0800 026 00 00 (kostenfrei).
- KEINE eigene Quellenliste generieren – die Quellen fügt das System am Ende an.
""").strip()

def _format_contexts(ctxs: list[str], max_chars_per_ctx: int = 1200, max_ctx: int = 4) -> str:
    # trim, dedupe, cap
    seen, out = set(), []
    for c in ctxs:
        c = (c or "").strip()
        if not c:
            continue
        h = hash(c[:200])  # cheap near-dup filter
        if h in seen:
            continue
        seen.add(h)
        out.append(c[:max_chars_per_ctx])
        if len(out) >= max_ctx:
            break
    return "\n\n---\n\n".join(out)


def build_prompt(question: str, contexts: list[str]) -> str:
    """Return a single string prompt for plain-text LLMs (Ollama)."""
    ctx_block = _format_contexts(contexts)
    user_block = dedent(f"""
    Verwende ausschließlich die folgenden Kontexte, um die Frage zu beantworten.
    Wenn keine eindeutige Antwort möglich ist, sage das explizit.

    [KONTEXTE]
    {ctx_block}

    [FRAGE]
    {question}

    Formatiere die Antwort kurz und verständlich. Erzeuge KEINE eigene Quellenliste.
    """).strip()
    # For simple LLMs we concatenate a system + user pattern.
    return f"[SYSTEM]\n{SYSTEM_DE}\n\n[USER]\n{user_block}\n\n[ASSISTANT]"