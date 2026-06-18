# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Guardianity — Kioku v1 · Researcher
"""Kioku Researcher — the memory engine, turned into an autonomous researcher.

One question → ~20 deep sub-questions → live web research on each → one complete,
uncut report, with every step remembered in a Kioku mind so the whole run can be
recalled and questioned afterwards.
"""
from engine.research.persistence import ResearchDB
from engine.research.researcher import DEFAULT_NUM_QUESTIONS, Finding, Researcher
from engine.research.runs import RESEARCH_TENANT, ResearchRun, RunManager
from engine.research.websearch import Source, WebSearch

__all__ = [
    "Researcher",
    "Finding",
    "DEFAULT_NUM_QUESTIONS",
    "WebSearch",
    "Source",
    "ResearchRun",
    "RunManager",
    "ResearchDB",
    "RESEARCH_TENANT",
]
