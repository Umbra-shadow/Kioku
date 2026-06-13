# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Guardianity — Kioku v1
"""The curiosity loop (§3.4) — Kioku's differentiator.

Like a human who hears a new word and quietly looks it up: for every term
in a fresh engram that Kioku has never seen (one shift+mask lookup against
the lexicon), it asks Qwen for a 1-2 sentence definition *in this user's
context* and locks it into the engram and the global lexicon.

Budget-capped and designed to run as a fire-and-forget task — it must
never block the reply.
"""

from __future__ import annotations

import logging
from typing import Callable

from engine.decompose import EventSink, PipelineEvent, _emit
from engine.engram import Engram, normalize_term
from engine.qwen import LLMError, QwenClient

log = logging.getLogger("kioku.curiosity")

DEFINE_SYSTEM = (
    "You are the curiosity stage of Kioku, a memory engine. Define the given "
    "term in 1-2 short sentences, grounded in the conversational context "
    "provided. If it is a person/place/product the user mentioned, say what "
    "it is to them. Plain text, no preamble."
)


async def curiosity_pass(
    qwen: QwenClient,
    engram: Engram,
    is_known: Callable[[str], bool],
    *,
    max_lookups: int = 3,
    emit: EventSink | None = None,
) -> dict[str, str]:
    """Research unknown terms; returns {term: definition}, already written
    into ``engram.definitions``. Failures are logged and skipped — partial
    curiosity beats a blocked pipeline."""
    unknown = [t for t in engram.index_terms() if not is_known(t)][: max(0, max_lookups)]
    learned: dict[str, str] = {}
    for term in unknown:
        term = normalize_term(term)
        try:
            definition = await qwen.chat(
                [
                    {"role": "system", "content": DEFINE_SYSTEM},
                    {
                        "role": "user",
                        "content": f'TERM: "{term}"\nCONTEXT: {engram.meaning or engram.message[:300]}',
                    },
                ],
                temperature=0.2,
                max_tokens=120,
            )
        except LLMError as e:
            log.warning("curiosity lookup failed for %r: %s", term, e)
            continue
        definition = definition.strip()
        if not definition:
            continue
        learned[term] = definition
        engram.definitions[term] = definition
        await _emit(emit, PipelineEvent("curious", engram.engram_id, {"term": term}))
    if learned:
        log.info("curiosity learned %d term(s) for %s", len(learned), engram.engram_id)
    return learned
