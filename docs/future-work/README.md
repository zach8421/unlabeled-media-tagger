# Future Work

Design notes and forward-looking thinking for features the project will
likely want, but isn't ready to implement yet. Not a roadmap — these
docs predate decisions, not commit to them.

## Purpose

These are working documents for ideas that need to mature before they're
worth coding. Drafting them now serves three purposes:

- Captures intent and reasoning while it's fresh, so the rationale survives
  past whoever's currently in the headspace.
- Surfaces design questions early enough that user testing or later
  decisions can answer them, rather than discovering the questions late
  when they're expensive to answer.
- Gives the team a place to push back, add context, or kill ideas before
  they get built.

A doc landing here is not a commitment to build the thing. Some of these
will become real slices; some will be replaced by better ideas; some will
be intentionally abandoned with the doc kept as the record of why.

## Conventions

Each document in this folder should:

- State its status at the top — `design notes`, `spec`, `abandoned`, etc.
- Describe the problem before the solution.
- Surface known design questions explicitly rather than assuming answers.
- Note what would have to be true (e.g., user-test results, sponsor
  decisions) before the work makes sense to start.

When a future-work doc graduates to active development, leave the doc in
place but link to the implementation slice prompts and any specs that
superseded it. The history is useful.

## Current documents

- [`feedback-loop.md`](feedback-loop.md) — User-driven cluster labeling,
  identity persistence across runs, and how human corrections improve
  future clustering. The largest planned post-user-test workstream.

## Adding a new document

Write what you're thinking, even if it's rough. A messy first draft beats
a polished idea that lives only in someone's head or in a chat log. Other
team members can sharpen the draft over time. The cost of a stale or
half-formed doc here is low; the cost of forgetting why a decision was
made is high.

If a doc reaches a state where it could become a real slice prompt, that's
the signal to move it out of `future-work/` and into whatever the active
development workflow uses.
