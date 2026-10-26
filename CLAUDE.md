# Accel-Sim study repo

- Personal study copy of Accel-Sim, published as `BeenLi/accel-sim-learning` (public, not a fork). Never sync with or contribute to upstream.
- Single branch `main`. Commit after each learning step (one note or one outline change) and push. GitHub Actions are off until our own CI is written (stage 01 todo).
- Upstream base is `c5e0872`; `git diff c5e0872 -- <path>` shows local changes. GPGPU-Sim vendoring: see `.claude/rules/accel-sim.md`.
- `README.md` describes this repo; upstream's README is `README-accel-sim.md`.

## Study notes (`docs/learning/`)

- Write them with the repo-learning-path skill (https://github.com/BeenLi/arch-reading-skills/tree/main/skills/repo-learning-path); naming, links, figures, link check and index sync all follow it.
- Before writing, read `docs/learning/README.md` §1–2 for the reader's background, references, markers (✅/❓ only, no 🆕) and note structure (concept → code → verify).
- Textbook (Aamodt et al. 2018) is local at `docs/learning/supplements/`: the PDF plus `gpgpu-arch-book.txt` for grep; PDF page = book page + 20. It is copyrighted and gitignored: never commit it, never copy its figures into `figs/`, quote only a sentence or two; cite as "教材 §3.3（p.35）".
