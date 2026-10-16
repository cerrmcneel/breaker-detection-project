# Working mode for this repo

**Agentic coding (since 2026-10-16).** This project started in a mentor mode,
where I wrote the code and the agent explained and reviewed. I've moved to full
agentic coding: agents write the implementation, and my job is orchestrating
the build: deciding what to build, reviewing results and approving what ships.

What that means for an agent working here:

1. Implement features and fixes directly, with tests, and verify them against
   the running system before calling them done (see the roadmap's invariants).
2. Decisions that are mine stay mine: product and scoring rules, regulatory
   statements, anything legal, and every outward-facing step (deploying,
   pushing the Marjal site, sending anything). Ask; don't assume.
3. Several agent sessions work in this repo at once. Check `git log`,
   open PRs and the roadmap before starting, so two sessions don't build the
   same thing.
4. Explain what you did in plain words, so I can review it without reading
   every line.

## Where the plan lives — read before starting work

`directives/production_hardening_roadmap.md` is the living plan: phase
status, what's done, what's next, and a **CURRENT STATE** section at the
top (a verified snapshot, then the prioritised open work) with uncommitted work, the deployment topology, and gotchas that will
bite you (notably: `C:\ironhack\labs\marjal-website` is a stale,
commit-less snapshot — pushing from it would destroy published work).

`directives/` is **gitignored**, so the roadmap, business plan, Hackshow prep
and presentation notes are local-only and won't appear in a fresh clone. (Six
older files, the principles/workflow/glossary/mentor docs and the K3s write-up,
were committed before the ignore rule and are still tracked.) Read it anyway if
it's present; update the phase status and
the CURRENT STATE section whenever you finish a meaningful chunk, so the
next session — human or agent — starts from the truth rather than from
this file's aspirations.

Known-good invariants worth not rediscovering the hard way:
- **Do not wire SAHI back into `src/model/pipeline.py`.** It is installed
  but deliberately unused; an ablation showed it hurt accuracy once the
  label-permutation bug was fixed. See the comment in
  `requirements-training.txt` (it moved there with the Phase 3 requirements split).
- **Production inference does not run on K3s today.** The GPU worker is a native
  Windows process started by the `PanelSafeInference` Scheduled Task
  (`scripts/start_inference.ps1`); `yolo-inference-deployment.yaml` is the May 2026
  K3s setup. `curl http://localhost:8088/` shows what is actually being served.
- **OCR is on in production** (since `prod/2026-09-29-release`): `use_ocr: true` in
  `src/model/pipeline_config.json`, independent of `use_hmm` (still `false`). Known
  confidently-wrong reads exist (a C40 read as `C4`), so ratings always need human review.
- **The GPU worker runs from the main working tree** (`C:\ironhack\labs\breaker-detection-project`,
  detached at the worker's release commit). Whatever is checked out there goes live at the
  worker's next restart, so never switch branches in it: work in a `git worktree`. Its
  `CLAUDE.md` is therefore the release-time copy; the current one is on `origin/main`.
- **Gateway deploys are git tags** `prod/YYYY-MM-DD-<name>`, checked out detached on VM 101
  (Tailscale `100.103.11.109`). `app/frontend` is bind-mounted (live on checkout); changes
  under `app/` or `src/` need `docker compose up -d --build web`. `requirements.txt` is
  pinned to the live image's versions; upgrade them only on purpose. cloudflared runs with
  `--protocol http2` (committed 2026-10-13; QUIC fails on that LAN). **Tags older than
  `prod/2026-10-13-compose-http2` carry the old QUIC file**, so after checking one out on the
  VM, restore it with `git checkout prod/2026-10-13-compose-http2 -- docker-compose.yml`
  before any `docker compose up`, or the public sites go down. Commands and rollbacks:
  `C:\ironhack\labs\panelsafe-ops\README.md` (outside the repo on purpose).
- **`/metodologia/` is stamped with a hash of `grade_panel_layout`'s source.** Any edit to
  that function, even formatting, must restamp `<code id="score-version">` in
  `app/frontend/metodologia/index.html` in the same commit
  (`src/tests/test_score_methodology.py` fails otherwise).
- The full test suite is expected to be **green** (`python -m pytest
  src/tests/ -q`). If something fails on arrival, it's a regression or
  rot, not the normal state.
