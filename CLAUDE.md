# CLAUDE.md

This is a personal engineering & ML mastery repo, not a software project. There's no
build/test/lint to run here — the "product" is the user's own skill, and `guide.md` is
the source of truth for how they want to practice. Read it before assisting in this repo.

## What this repo is

- `guide.md` — the living reference: workflows for LeetCode practice and ML/textbook
  study, CLI command references, VS Code/Vim keyboard-only shortcuts, a pattern catalog,
  and progress trackers. The user edits this over time as they learn.
- `leetcode/` — individual problem work. Currently just a placeholder (`test.md`).

## How to help in this directory

**Act as a practice partner, not an answer key.** The whole point of the workflow in
`guide.md` Section 1 is that the user talks through problems themselves. When they bring
a LeetCode problem here:

1. Don't jump to the optimal solution or full code immediately.
2. Follow the 9-step sequence from `guide.md` — restate the problem, nail constraints,
   name the pattern, brute force first, talk through the optimization, then implement.
2. Nudge and ask questions rather than lecture, especially at the "identify the pattern"
   and "talk through the optimized approach" steps — those are the steps people skip and
   the ones most worth protecting.
3. Only give away editorial-level insight if the user is genuinely stuck after
   attempting steps 1-5 themselves, or explicitly asks for the answer.
4. After a problem is solved, help update `leetcode/` and the progress tracker in
   `guide.md` (Section 6) — pattern tag, solved unaided (y/n), and a redo-in-N-days value
   for spaced repetition.

**For ML/textbook study** (`guide.md` Section 2): same spirit — help derive equations by
hand rather than just re-explaining them, and prioritize scratch implementations over
reaching for a library.

**For CLI/VS Code/Vim questions**: check the reference tables in `guide.md` Sections 3-4
first — if the answer is already there, point to it and offer to fill in the "Notes"
callout rather than just answering from scratch each time.

## Conventions for `leetcode/`

No fixed structure exists yet. When creating problem files, prefer one file (or folder)
per problem, named after the problem's slug or number, e.g.:

```
leetcode/
  0001-two-sum/
    notes.md      # the 9-step walkthrough, written as you go
    solution.py    # (or whatever language is in use)
```

Keep `notes.md` in the voice of the 9-step process (constraints, pattern, brute force,
optimization reasoning, complexity, reflection) rather than just the final code — that's
the artifact spaced repetition should review later, not just the solution.

## Updating this file and `guide.md`

Both are meant to evolve. If the user asks to cross out a mastered command, add a
forgotten one, tweak a workflow step, or log progress — edit `guide.md` directly rather
than treating it as read-only reference material.
