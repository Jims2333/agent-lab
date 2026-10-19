"""
Rebuild 05-llm-and-api/data/corpus.txt, the training text for Chapter 5's labs.

  python tools/build_corpus.py          # rewrites the file in place

The corpus is the body text of the READMEs of Chapters 1-4, as plain prose:
  * line 1 is a comment (starting with "#") that describes the corpus; the labs skip it
  * then, for each chapter, its title line followed by its prose, one line per line of text
  * removed: fenced code blocks, tables, horizontal rules, blank lines, images and HTML tags,
    link URLs (the link text is kept), and everything from "## References" on (which also
    drops the footer navigation)
  * stripped: heading, blockquote and list markers, Markdown emphasis (* and **), inline-code
    backticks (the code itself is kept), and the 💡 🧪 📖 call-out emoji
  * very short lines (fewer than MIN_CHARS characters, e.g. "Labs" or "Concepts") are dropped
Run it again whenever one of those READMEs changes, then re-run the Chapter 5 labs.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHAPTERS = ["01-ml-origins", "02-linear-models", "03-neural-nets", "04-cnn-to-transformer"]
OUT = ROOT / "05-llm-and-api" / "data" / "corpus.txt"
HEADER = ("# Corpus: the body text of the READMEs of Chapters 1–4 of this repo "
          "(tables, code, links and references removed), used for the labs in Chapter 5.")
MIN_CHARS = 9

FENCE = re.compile(r"^```.*?^```\n?", re.S | re.M)
URL = r"\((?:[^()\s]|\([^()\s]*\))*\)"               # a link target, allowing one level of (...)
IMAGE = re.compile(r"!\[[^\]]*\]" + URL)
LINK = re.compile(r"\[([^\]]*)\]" + URL)
HTML = re.compile(r"<[^>]+>")
EMPHASIS = re.compile(r"\*+(?=\S)|(?<=\S)\*+")     # * or ** touching a word; a lone " * " stays
EMOJI = re.compile("[💡🧪📖]")


def clean(line):
    s = line.strip()
    s = re.sub(r"^#+\s*", "", s)                     # heading markers
    s = re.sub(r"^>\s?", "", s).strip()              # blockquote marker
    s = re.sub(r"^[-*]\s+", "", s)                   # bullet-list marker
    s = IMAGE.sub("", s)
    s = LINK.sub(r"\1", s)
    s = HTML.sub("", s)
    s = EMPHASIS.sub("", s).replace("`", "")
    s = EMOJI.sub("", s)
    return re.sub(r"\s+", " ", s).strip()            # also turns the ideographic space into " "


def chapter_text(path):
    text = FENCE.sub("", path.read_text(encoding="utf-8"))
    lines = []
    for raw in text.splitlines():
        s = raw.strip()
        if s.startswith("## References"):
            break
        if not s or s.startswith("|") or s == "---":
            continue
        s = clean(s)
        if len(s) >= MIN_CHARS:
            lines.append(s)
    return lines


def main():
    lines = [HEADER]
    for chapter in CHAPTERS:
        lines += chapter_text(ROOT / chapter / "README.md")
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    body = "\n".join(lines[1:]) + "\n"                 # what the labs read: everything but line 1
    print(f"wrote {OUT.relative_to(ROOT)}: {len(lines)} lines, {len(body):,} characters")


if __name__ == "__main__":
    main()
