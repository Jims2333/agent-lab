"""
Check every relative link and anchor in the repo's Markdown files.

  python tools/check_links.py          # exit code 1 if anything is broken

What it checks:
  * [text](path) and [text](path#anchor) links to files inside the repo
  * [text](#anchor) links to headings in the same file
  * src="..." / srcset="..." paths in inline HTML (images, <picture> sources)
Anchors are computed the way GitHub does it: lowercase, punctuation removed,
spaces turned into hyphens, and "-1", "-2", ... appended to repeated headings.
Links inside fenced code blocks are ignored. External http(s) links are not fetched.
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FENCE = re.compile(r"^```.*?^```", re.S | re.M)
LINK = re.compile(r"\]\(([^)\s]+)\)")
HTML_SRC = re.compile(r"\b(?:src|srcset)=\"([^\"]+)\"")
HEADING = re.compile(r"^#{1,6} (.+?)\s*#*\s*$", re.M)


def slug(heading):
    return re.sub(r"[^\w\- ]", "", heading.strip().lower()).replace(" ", "-")


def anchors(path):
    text = FENCE.sub("", path.read_text(encoding="utf-8"))
    seen, out = {}, set()
    for m in HEADING.finditer(text):
        s = slug(m.group(1))
        n = seen.get(s, 0)
        out.add(s if n == 0 else f"{s}-{n}")
        seen[s] = n + 1
    return out


def check(md):
    problems = []
    text = FENCE.sub("", md.read_text(encoding="utf-8"))
    targets = [(m.group(1), True) for m in LINK.finditer(text)]
    targets += [(m.group(1), False) for m in HTML_SRC.finditer(text)]
    for link, may_have_anchor in targets:
        if re.match(r"^[a-z]+:", link):            # http:, https:, mailto: ...
            continue
        path, _, frag = link.partition("#") if may_have_anchor else (link, "", "")
        target = (md.parent / path).resolve() if path else md.resolve()
        if path and not target.exists():
            problems.append(f"missing file   {md.relative_to(ROOT)} -> {link}")
            continue
        if frag:
            page = target / "README.md" if target.is_dir() else target
            if page.suffix != ".md" or frag not in anchors(page):
                problems.append(f"missing anchor {md.relative_to(ROOT)} -> {link}")
    return problems


def main():
    problems = []
    for md in sorted(ROOT.rglob("*.md")):
        if ".git" in md.parts:
            continue
        problems += check(md)
    for p in problems:
        print(p)
    print(f"link problems: {len(problems)}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
