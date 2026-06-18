# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Guardianity — Kioku v1
"""Test-suite guards — keep tests OFF live credentials.

``settings()`` loads ``.env`` (which may carry a real ``DATABASE_URL`` for Neon
Postgres and a real ``QWEN_API_KEY``). Tests must never touch those: they run on a
fake brain and an ephemeral in-memory database. Setting these at conftest import
time — before any fixture calls ``settings()`` — forces the durable store to an
in-memory SQLite and drops any inherited cloud DSN, so no test ever reaches the
network or the operator's real database.
"""
import os

# Force every app the suite builds onto an ephemeral, local, in-memory store.
os.environ["DATABASE_URL"] = ""          # never connect to Neon/Postgres in tests
os.environ["KIOKU_RESEARCH_DB"] = ":memory:"
os.environ.setdefault("KIOKU_RATELIMIT", "off")
