#!/usr/bin/env python3
"""Turn each chapter folder's card files into one flashcard deck.

Expected layout — one folder per chapter:

    textbook/
      hands_on_ml/
        ch01-machine-learning-landscape/
          notes.ipynb       <- your notes (ignored here)
          exercises.md      <- the book's end-of-chapter questions
          concepts.md       <- extra cards covering the chapter text

Either card file is optional; a folder with at least one becomes a deck, with
exercise cards first, then concept cards. Only chapter folders count (one level
below a book folder): a card file anywhere else, like a book-level list of every
exercise, is skipped. Both files use the same format:

    # Chapter 1 - The Machine Learning Landscape   <- deck title (first H1)

    ## 1. How would you define machine learning?   <- card front
    Machine Learning is about building systems...  <- card back (until next ##)

The leading "1." is optional; cards are numbered by position when it's absent.
Every "##" heading becomes a card, so keep stray sections out of these files (or
demote them to "###", which is treated as part of the answer body). Text before
the first "##" is ignored, so it's a good place for notes about the file.

Usage:
    python3 textbook/flashcards/build_cards.py

Writes textbook/flashcards/cards.js, which index.html loads directly (a plain
script tag, so opening the page as a file:// URL works without a web server).
"""

import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TEXTBOOK_DIR = REPO_ROOT / "textbook"
OUTPUT = Path(__file__).resolve().parent / "cards.js"

# (filename, card kind, key prefix) — decks list exercise cards before concept cards.
CARD_FILES = [
    ("exercises.md", "exercise", "e"),
    ("concepts.md", "concept", "c"),
]

TITLE = re.compile(r"^#\s+(.+?)\s*$", re.MULTILINE)
LIST_ITEM = re.compile(r"^\s*(?:[-*•]|\d{1,2}[.)])\s+")
# A card front: an H2 heading, with an optional "12." prefix we strip for display.
CARD_HEADING = re.compile(r"^##\s+(?:(\d{1,3})\.\s*)?(.+?)\s*$", re.MULTILINE)


def normalize(text):
    """Unwrap hard-wrapped prose, keeping paragraph breaks and list items.

    A wrapped line is joined onto the one above it, unless it starts a list
    item ("- ", "* ", "1. "), which stays on its own line.
    """
    paragraphs = []
    for block in re.split(r"\n\s*\n", text.strip()):
        lines = []
        for raw in block.splitlines():
            line = " ".join(raw.split())
            if not line:
                continue
            if lines and not LIST_ITEM.match(raw):
                lines[-1] += " " + line
            else:
                lines.append(line)
        if lines:
            paragraphs.append("\n".join(lines))
    return "\n\n".join(paragraphs)


def parse_cards(path, kind, prefix):
    text = path.read_text()
    headings = list(CARD_HEADING.finditer(text))

    cards, empty = [], []
    for i, heading in enumerate(headings):
        end = headings[i + 1].start() if i + 1 < len(headings) else len(text)
        number = int(heading.group(1)) if heading.group(1) else i + 1
        answer = normalize(text[heading.end():end])
        if not answer:
            empty.append(number)
            continue
        cards.append({
            "key": f"{prefix}{number}",
            "n": number,
            "kind": kind,
            "q": normalize(heading.group(2)),
            "a": answer,
        })

    if empty:
        print(f"  {path.name}: {len(empty)} question(s) with no answer yet, skipped: {empty}")

    keys = [c["key"] for c in cards]
    duplicates = sorted({k for k in keys if keys.count(k) > 1})
    if duplicates:
        sys.exit(f"  {path}: duplicate card numbers {duplicates} — progress tracking needs them unique")

    title = TITLE.search(text)
    return cards, (title.group(1) if title else None)


def build_deck(folder):
    cards, title, sources = [], None, []
    for filename, kind, prefix in CARD_FILES:
        path = folder / filename
        if not path.exists():
            continue
        file_cards, file_title = parse_cards(path, kind, prefix)
        print(f"  {filename}: {len(file_cards)} cards")
        cards += file_cards
        title = title or file_title
        sources.append(path.relative_to(REPO_ROOT).as_posix())

    if not cards:
        return None

    relative = folder.relative_to(REPO_ROOT).as_posix()
    return {
        "id": re.sub(r"[^a-z0-9]+", "-", relative.lower()).strip("-"),
        "title": title or folder.name,
        "sources": sources,
        "cards": cards,
    }


def chapter_order(folder):
    # Appendix folders ("appA-...") would otherwise sort ahead of "ch01-...".
    return (folder.parent.as_posix(), folder.name.startswith("app"), folder.name)


def chapter_folders():
    """Folders at textbook/<book>/<chapter>/ that hold at least one card file."""
    folders = set()
    for name, _, _ in CARD_FILES:
        for path in TEXTBOOK_DIR.rglob(name):
            if len(path.relative_to(TEXTBOOK_DIR).parts) == 3:
                folders.add(path.parent)
            else:
                print(f"skipped {path.relative_to(REPO_ROOT)}: not in a chapter folder")
    return sorted(folders, key=chapter_order)


def main():
    folders = chapter_folders()
    if not folders:
        sys.exit(
            f"no card files found under {TEXTBOOK_DIR}\n"
            f"expected e.g. textbook/<book>/ch01-<slug>/exercises.md or concepts.md"
        )

    decks = []
    for folder in folders:
        print(folder.relative_to(REPO_ROOT))
        deck = build_deck(folder)
        if deck:
            decks.append(deck)

    if not decks:
        sys.exit("no decks built")

    payload = json.dumps(decks, indent=2, ensure_ascii=True)
    OUTPUT.write_text(
        "// Generated by build_cards.py - edit the chapter .md files, then re-run it.\n"
        f"window.DECKS = {payload};\n"
    )
    total = sum(len(d["cards"]) for d in decks)
    print(f"\nwrote {OUTPUT.relative_to(REPO_ROOT)}: {len(decks)} deck(s), {total} cards")


if __name__ == "__main__":
    main()
