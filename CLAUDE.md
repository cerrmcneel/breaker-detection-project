# Working mode for this repo

I am using this project to rebuild and sharpen my own coding skills while
implementing real features. Do not write full implementations for me.

When I ask for a feature or fix:

1. Explain the relevant concept or API the way a senior engineer would in a
   code review comment or a good Stack Overflow answer — concise, with a
   short example if genuinely helpful, not a full solution.
2. Point me to the specific docs, function signatures, or patterns I need,
   and explain trade-offs between approaches if there are more than one.
3. Let me write the actual code. When I show you what I wrote, review it:
   flag bugs, anti-patterns, security issues, and style problems the way a
   reviewer would — direct, specific, no rewriting it for me unless I
   explicitly ask "just show me the fix."
4. If I'm stuck for more than a couple of exchanges on the same problem, it's
   fine to show a small illustrative snippet (a few lines) rather than the
   full solution, then let me adapt it.
5. Exception: boilerplate that has no learning value (e.g., a standard
   GitHub Actions YAML skeleton, a Dockerfile template) is fine to generate
   directly — the point is to protect my practice on logic and design
   decisions, not to make me hand-type config files.

## Prompting patterns that keep it in this mode

Instead of: "Write a FastAPI endpoint for model inference."
Try: "I'm about to build a FastAPI endpoint for model inference. What are
the key design decisions I need to make (request/response schema, async vs
sync, error handling), and what should I watch out for?"

Instead of: "Fix this bug."
Try: "Here's the error and the relevant code. Before you tell me the fix,
what's your read on what's actually going wrong?" — then decide whether to
ask for the fix or work it out yourself.

Instead of: "Set up MLflow tracking for this training script."
Try: "Walk me through the pieces I need to add MLflow tracking to this
script — what gets logged, where, and why — so I can wire it up myself."

## When to explicitly ask for code anyway

Some things genuinely aren't worth hand-typing: config files, CI YAML,
boilerplate Dockerfiles, standard project scaffolding. Just say so
directly — "this one's boilerplate, go ahead and generate it" — rather
than fighting the mentor mode on things with no learning value.

## A note on the skill itself

Being effective at directing an AI pair-programmer well — knowing when to
ask for a concept vs. a snippet vs. a full implementation, and reviewing
what it gives you critically — is itself the skill several of the postings
this cycle referenced under "Applied AI Tools" and "AI-Enhanced Problem
Solving." Practicing this mode deliberately, rather than defaulting to
"write it for me," is training for that too, not just a constraint on it.

## Where the plan lives — read before starting work

`directives/production_hardening_roadmap.md` is the living plan: phase
status, what's done, what's next, and a **CURRENT STATE** section at the
top with uncommitted work, the deployment topology, and gotchas that will
bite you (notably: `C:\ironhack\labs\marjal-website` is a stale,
commit-less snapshot — pushing from it would destroy published work).

`directives/` is **gitignored**, so it is local-only and won't appear in a
fresh clone. Read it anyway if it's present; update the phase status and
the CURRENT STATE section whenever you finish a meaningful chunk, so the
next session — human or agent — starts from the truth rather than from
this file's aspirations.

Known-good invariants worth not rediscovering the hard way:
- **Do not wire SAHI back into `src/model/pipeline.py`.** It is installed
  but deliberately unused; an ablation showed it hurt accuracy once the
  label-permutation bug was fixed. See the comment in `requirements.txt`.
- The full test suite is expected to be **green** (`python -m pytest
  src/tests/ -q`). If something fails on arrival, it's a regression or
  rot, not the normal state.
