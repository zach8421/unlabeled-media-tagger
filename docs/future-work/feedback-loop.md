# Feedback Loop — Design Notes

> Status: design notes, not a spec. Captures intent and known design
> decisions for a future iteration of the pipeline. Do not implement
> without first running a user test against the current pipeline and
> updating this doc with what's learned.

## The problem this solves

Face clustering on real-world media doesn't produce one cluster per person.
A single person ends up split across many machine-generated cluster IDs
because their appearance varies — different angles, profiles, lighting,
expressions, distance from camera, occlusion. The pipeline can't fix this
on its own; the embedding model has fundamental limits on what it considers
"the same face."

Humans can fix it trivially. A user looking at two clusters labeled
person_007 and person_142 can tell instantly that they're both Sarah, even
when the embedding similarity score doesn't quite reach the merge threshold.

This document describes a feedback loop that captures user judgments,
applies them to the current data, and uses them to improve future
clustering runs.

## Use case context

The end user is a media company (Converge). They produce interview-driven
content. The faces they care about are people being interviewed, featured,
or otherwise editorially significant. Their own staff, crew, and
incidental background faces appear constantly in footage but aren't worth
tracking by name — the user wants control over which identities populate
their working spreadsheet without the pipeline forgetting those identities
exist.

This shapes several design decisions below. In particular, "untracked but
known" must be a first-class state, distinct from both "labeled" and "never
seen."

## User-facing behavior

### Identity tabs

The spreadsheet UI gains per-identity views — one tab per labeled person.
Sarah's tab shows every face/file currently assigned to Sarah, sourced from
every cluster that's been labeled or merged into Sarah's identity. The user
can:

- Browse all of Sarah's appearances without searching Drive directly. This
  is a discovery affordance — finding "the shot of Sarah from the mayoral
  debate" becomes a sort/filter operation, not a hunt.
- Spot-check the assignments. If a face on Sarah's tab isn't actually
  Sarah, the user can correct it.

### Correction actions on a misassigned face

When the user finds a face on Sarah's tab that isn't Sarah, they have three
actions, expressed via a dropdown or similar UI element on that row:

1. **Reassign to a known identity.** Select another labeled person from
   the dropdown. The face moves from Sarah's identity to that person's.
2. **Unassign / mark blank.** The face is no longer attached to any
   labeled identity. It returns to being a member of its original machine
   cluster (if still meaningful) or floats free.
3. **Mark as untracked.** The face is associated with a labeled identity,
   but that identity is flagged "do not write back to the source spreadsheet."
   This is for staff, crew, hosts, or others the user knows about but
   doesn't want appearing in the working asset log.

The third action is the subtle one. It's not "delete this person." It's
"keep modeling this person internally — so the next time their face shows
up the system recognizes them and doesn't pollute someone else's cluster —
but don't surface them to me."

### Untracked vs. unlabeled

These must be different states, both in storage and in UI:

- **Unlabeled**: machine-generated cluster, no human has touched it. Shows
  up in review queues. No name. Default state.
- **Labeled, tracked**: human has named this person and wants them in the
  working spreadsheet. Writeback to Drive descriptions enabled.
- **Labeled, untracked**: human has named this person and wants the
  pipeline to remember them, but excluded from the working spreadsheet.
  Writeback can be enabled or disabled — likely disabled by default for
  this state, but configurable per identity.

A column like `tracked` or `writeback` on the identity record captures
this. `tracked = false` means the identity exists, the model uses it for
clustering and re-identification, but the spreadsheet's day-to-day views
filter it out.

## Data model

The current pipeline output has an implicit data model: cluster IDs and
the faces in each cluster, regenerated fresh on every run. That model has
no concept of persistent identity, so it can't survive across runs and
can't accumulate user feedback.

The new model needs three layers:

### Layer 1 — Faces (already exists)

Individual face detections. Each has an embedding, source file, frame,
bbox, confidence. The atomic unit. Owned by the pipeline. Regenerated each
run from source media.

### Layer 2 — Clusters (already exists, but re-scoped)

Machine-generated groupings of faces by embedding similarity. Owned by the
pipeline. Regenerated each run. Should *not* be treated as stable
identities — cluster IDs may change between runs as new media is added or
clustering parameters change.

This is important: the current `person_007` is a *cluster label*, not an
*identity*. Users should not be expected to remember cluster IDs across
runs.

### Layer 3 — Identities (new)

Persistent named entities. Owned by the user's labeling decisions. Each
identity has:

- A stable internal ID (e.g., `identity_001`, generated once and never
  reassigned)
- A display name (e.g., "Sarah Chen") — editable, can be blank
- A `tracked` flag (boolean) — whether this identity appears in the
  working spreadsheet
- A `writeback` flag (boolean) — whether to write this name to Drive file
  descriptions
- A list of cluster IDs from any number of past runs that have been
  labeled with this identity
- A list of explicit per-face overrides (face X belongs to this identity
  even though its cluster doesn't)
- An audit trail: who labeled what, when, and why (see "Reversibility"
  below)

The relationship is many-to-one: many cluster IDs can map to one identity.
A new run produces new cluster IDs; the system attempts to auto-attach
them to existing identities based on centroid similarity, with thresholds
and user confirmation as appropriate.

## Reversibility

Every user action must be reversible. Without this, mistakes compound:
the user labels person_007 as Sarah, later realizes person_007 contained a
false positive, and now Sarah's centroid is permanently corrupted.

Reversibility implications:

- **Don't destroy data on merge.** When the user assigns cluster
  person_007 to identity Sarah, *do not* delete person_007 or merge its
  embeddings into a single Sarah-blob. Keep the cluster intact and the
  embeddings as they were. The identity is a *reference layer*, not a
  destructive operation.
- **Audit trail for every assignment.** Every "this cluster belongs to
  this identity" link is its own record with a timestamp. Undo means
  deleting that record, which restores the cluster to whatever state it
  was in before.
- **No automatic centroid replacement.** Sarah's "centroid" for purposes
  of matching new faces is computed *on demand* from her current
  assignments, not stored as a separate object that can drift out of sync.
  Recompute is cheap; storing a derived value that can become inconsistent
  with its source is expensive in subtle ways.
- **Spot-check correction is just another assignment.** When the user says
  "this face on Sarah's tab isn't actually Sarah, it's Marcus," that's a
  per-face override that overrides the cluster-level assignment. Removing
  the override restores the cluster-level assignment.
- **Three undo levels.** Per-face override (smallest), cluster-to-identity
  assignment (medium), full identity record (largest, with cascade
  warning). Each should be individually reversible.

## How feedback improves clustering

This is the part that has to be designed carefully. Naive approaches drift.

### What helps

- **Identity-aware clustering on subsequent runs.** When a new run
  produces a new cluster, compare its centroid not just to other new
  clusters but to all existing labeled identities' implicit centroids.
  High-confidence matches auto-attach to the existing identity; medium
  confidence surfaces for user review; low confidence stays as a new
  unlabeled cluster.
- **Wider, more general identity centroids.** An identity with 50 faces
  spanning multiple angles and lighting conditions matches new faces of
  that person more reliably than a single-angle 3-face cluster does. This
  is the genuine machine learning win from the feedback loop. It happens
  automatically once layer 3 exists and is consulted during clustering.
- **Cannot-link constraints.** When the user reassigns a face from Sarah
  to Marcus, that's evidence that the underlying cluster shouldn't have
  contained both. Recording these can-not-link signals lets the clusterer
  avoid making the same mistake on similar new data.

### What doesn't help (and might hurt)

- **Naive merging that pulls centroids toward outliers.** If the user
  merges two clusters and one had a misclustered face in it, that wrong
  face is now influencing the identity's centroid. This is why the data
  model keeps faces and clusters intact rather than blob-merging
  embeddings.
- **Treating user labels as infallible ground truth in retraining.**
  Users mislabel, especially under time pressure. Confidence in a label
  should decay if it's not periodically reaffirmed, or if subsequent
  labels create contradictions.
- **Eager auto-tagging at low confidence.** The temptation is to use new
  identity centroids aggressively to label new clusters. Resist. False
  auto-tags train the user to distrust the system, which is a much harder
  problem to recover from than under-tagging.

## Open design questions

These need user-test data to answer well. Don't decide them in advance.

1. **Threshold for auto-attach.** When a new cluster matches an existing
   identity's centroid above some similarity threshold, does the system
   auto-attach silently, attach with a "needs review" flag, or just
   suggest? The right answer depends on observed false-positive rates from
   user testing.
2. **What does "unassign / mark blank" do to clustering signals?** If the
   user pulls a face out of Sarah, that face is now... what? A standalone
   identity? Attached back to its source cluster? Floating? This matters
   for what the next clustering run does with it.
3. **Are identities cross-project or per-project?** A media company might
   have hundreds of recurring people. Does Sarah persist across all
   projects, or is each project its own labeling space? Probably the
   former, but it's a real decision with security/privacy implications.
4. **How does the user discover that two existing identities are the same
   person?** They labeled person_007 as Sarah Chen yesterday and
   person_493 as Sarah today, not realizing they were the same. The system
   should surface likely-duplicate identities for review. Mechanism TBD.
5. **What's the merge UI for two identities?** Distinct from
   cluster-to-identity assignment. Almost certainly a separate
   workflow with a confirmation step.

## Implementation order (sketch, post-user-test)

This is a multi-slice effort. Rough shape, in order:

1. **Design slice.** Update this document based on user-test observations.
   Make the open-question decisions. Write schema definitions. No code.
2. **Schema slice.** Extend the data model to support layer 3 (identities)
   with persistent storage. Read-only at first — pipeline reads identity
   records but no UI exists to create them.
3. **Identity creation UI slice.** First version of the labeling
   workflow: name a cluster, save the identity, view the labeled tab.
   Reversible at the cluster-to-identity level. No spot checking yet, no
   auto-attach.
4. **Spot-check slice.** Per-face overrides on identity tabs. The dropdown
   actions described above. Per-face reversibility.
5. **Auto-attach slice.** Use identity centroids during clustering on
   subsequent runs. Surface medium-confidence matches for user review.
   This is when the feedback loop actually closes.
6. **Tracked / untracked slice.** The `tracked` and `writeback` flags as
   first-class concepts. Filter views accordingly. By this point we'll
   know from real usage whether these are two separate flags or one.

Each slice gets its own design conversation and prompt when it's time.

## Tedium curve

The first 50 labels will be slow. Expect this. The next 200 will be
faster because medium-confidence auto-attach reduces them to confirm-or-
reject decisions. By the time the system has 500 labeled identities with
broad coverage, marginal cost of labeling new media approaches zero.

The slope of that curve is the design quality of the feedback loop.
Centroid representation, threshold tuning, and UI clarity all directly
affect how many labels per minute the user can produce. Optimize for that
metric explicitly when designing the actual UI.

## Coordination with the spreadsheet UI

Most of what's described here is server-side / pipeline-side. The
spreadsheet UI (Apps Script + sheet structure) needs to surface it all.
Whoever owns the spreadsheet needs to be involved in design decisions for
slices 3+ — the data model implies sheet structure, and bad sheet
structure can sandbag a good data model. Keep them in the loop early.
