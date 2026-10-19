# Contributing to agent-lab

Thanks for helping! This is a course people learn from, so the most valuable contributions are the ones that make it
**more accurate** and **easier to follow**:

- **Fact corrections**: a date, name, number or quote that is wrong or unsupported. Please use the
  [fact correction](https://github.com/Jims2333/agent-lab/issues/new?template=fact_correction.yml) form and include a source.
- **Bug reports**: a lab that crashes, prints something confusing, or disagrees with its README.
- **Clearer explanations**: a paragraph or concept that took you three readings.
- **New ideas**: a story, lab or chapter that fits the course (open a
  [chapter idea](https://github.com/Jims2333/agent-lab/issues/new?template=chapter_idea.yml) first, so we can discuss it before you write it).

## The rules every chapter follows

**Stories are true.**
- Every name, date, number and quote must come from a source, and each chapter lists its sources under *References*.
- A detail that only one person reported is written as "according to X" or "as X later recalled".
- If something can't be verified, it is cut. We never invent dialogue.

**Story first, concepts last.**
- A chapter is a sequence of acts (time, place, people), each ending with a *Look ahead* pointer.
- Then come the labs, then a *Concepts* section that explains each idea and links back to the act where it first appeared, then *References*.

**Code is written from scratch.**
- Use only numpy or the standard library, so the idea stays visible.
- Fix random seeds, so the output is reproducible.
- End every script with a few *Think about it* questions.

**Every lab runs without an API key.**
- Labs that talk to a language model have a mock mode, and mock mode is the default.
- Real-model mode reads keys from environment variables and must never require them.

**Safety.**
- Never automate consumer chat apps.
- Never commit API keys.
- Labs that execute code or open ports must explain their sandboxing and permissions.

## Before you open a pull request

Run the same checks as CI, from the repository root:

```bash
python -m compileall -q .
python tools/run_labs.py        # runs every lab with the mock model
python tools/check_links.py     # every relative link and #anchor must resolve
```

If you changed a lab's output, update the "What you should see" table in that chapter's README so the numbers match.
If you added a fact, add its source to the chapter's references.

## Commit messages

Use a short imperative summary line, e.g. `Fix the Ceres orbit date in chapter 2`, followed by a blank line and a
description of what changed and why if it isn't obvious.

## Code of conduct

By participating you agree to follow the [Code of Conduct](./CODE_OF_CONDUCT.md).
By contributing you agree that your code is released under the [MIT license](./LICENSE) and your text under
[CC BY 4.0](./LICENSE-docs.md).
