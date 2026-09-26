# CLAUDE.md

This is a personal engineering & ML mastery repo, not a software project. There's no
build/test/lint to run here — the "product" is the user's own skill, and `guide.md` is
the source of truth for how they want to practice. Read it before assisting in this repo.

## What this repo is

- `guide.md` — the living reference: workflows for LeetCode practice and ML/textbook
  study, CLI command references, VS Code/Vim keyboard-only shortcuts, a pattern catalog,
  and progress trackers. The user edits this over time as they learn.
- `problems/` — individual problem work, one notebook per problem, grouped by source
  (`problems/leetcode/`, `problems/leetcode/75/`, `problems/bctci/`).
- `textbook/` — chapter-by-chapter study: notes, end-of-chapter exercises, and concept
  cards, plus a small flashcard app that turns them into one study deck per chapter.
  See the conventions below.

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

## Conventions for `problems/`

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

## Conventions for `textbook/`

One folder per book, one folder per chapter inside it:

```
textbook/
  flashcards/                          # the study app (below)
  hands_on_ml/
    ch01-machine-learning-landscape/
      notes.ipynb    # notes, worked examples, scratch implementations
      exercises.md   # the book's end-of-chapter conceptual questions + answers
      concepts.md    # extra cards on chapter content the exercises don't cover
    appA-autodiff/   # appendices use appX- and sort after the chapters
    handson-mlp-main/  # the author's repo (Apache 2.0): notebooks + exercise solutions
```

Keep the `chNN-` prefix so chapters sort in reading order. Notes can be split across
several notebooks in the folder if a chapter warrants it.  answers
come from the author's published solutions where they exist, and each file's intro
paragraph says where its answers came from. The book's coding exercises aren't cards;
each `exercises.md` intro lists them so they can be done in a notebook.

`exercises.md` and `concepts.md` use the same format — one `##` heading per question,
with the answer directly underneath:

    # Chapter 1 — The Machine Learning Landscape

    ## 4. What are the two most common supervised tasks?

    The two most common supervised tasks are regression and classification.

The first `#` heading becomes the deck title. The `4.` prefix is optional (cards number
by position without it). Every `##` becomes a card, so any other section in the file
should be `###` or lower; text before the first `##` (the intro paragraph) isn't a
card. Answer paragraphs are joined, but `-` / `1.` list lines are kept as lists. A
question with no answer under it yet is skipped with a warning, which makes it fine to
stub out a chapter's questions before writing answers.


### Flashcards

`textbook/flashcards/index.html` is a standalone study page — no server, no dependencies,
just open it in a browser. After adding or editing any `exercises.md` or `concepts.md`:

```
python3 textbook/flashcards/build_cards.py
```

That regenerates `textbook/flashcards/cards.js`; it's generated output, so don't
hand-edit it. Each chapter folder becomes one deck, with its `exercises.md` cards
(labelled "Book exercise N") followed by its `concepts.md` cards ("Concept N").
Study progress (got it / review again) is stored in the browser's localStorage, keyed
by the chapter folder path plus the card's number (`e4`, `c12`). So renaming or moving
a chapter folder resets that chapter's progress, and renumbering cards shuffles which
cards count as known. Add new cards at the end of a file rather than renumbering.

## Updating this file and `guide.md`

Both are meant to evolve. If the user asks to cross out a mastered command, add a
forgotten one, tweak a workflow step, or log progress — edit `guide.md` directly rather
than treating it as read-only reference material.
