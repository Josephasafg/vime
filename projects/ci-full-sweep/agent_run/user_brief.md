# User Brief

**Date**: 2026-06-08T02:10:16Z
**Worktree**: `ci-full-sweep`

## Verbatim ask

comprehensive CI sweep + slime A/B + bisect — pull latest vime (origin/main), run ALL CI tests on h200 with vime-latest; on any failure run the same test with the slime image as A/B to confirm a real issue; git-bisect regressions. Setup first.

---

> This file preserves the user's original task description verbatim. Never
> rewrite or paraphrase it. If scope changes mid-task, append a `decision`
> event to `events.jsonl` instead.
