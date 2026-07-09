"""Human-review loop: persistent label store + local review server.

Implements the minimum viable slice of docs/future-work/feedback-loop.md's
data model: a persistent identity layer (Layer 3) fed by an append-only event
log, with every action reversible. Clusters stay ephemeral per run; only
face_key- and identity-level records persist across re-clustering.
"""
