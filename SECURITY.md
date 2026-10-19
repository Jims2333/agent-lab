# Security policy

agent-lab is teaching material, not a production service, but a few labs do things worth being careful about:

| Where | What it does | What protects you |
|---|---|---|
| `07-harness/harness.py` | Lets an agent edit files and run tests | Works only inside a throwaway temporary folder; deleting files needs your approval |
| `08-multi-agent/write_review.py` | Runs code written by a model | Runs it in a subprocess with a timeout, CPU and memory limits, an empty working folder and no API keys in its environment. This is **not** a real sandbox. |
| `09-cross-model/a2a_mini.py`, `capstone.py` | Start small HTTP servers | Listen on `127.0.0.1` only and stop when the script ends |
| `side-quests/app-bridge/` | An MCP server that two desktop apps can reach, optionally through a tunnel | Read that side quest's safety notes before exposing anything to the internet |

## API keys

- Put keys in environment variables (or Codespaces secrets), never in code or in committed files.
- All labs run without any key; real-model mode is opt-in.
- If you accidentally commit a key, revoke it with the provider immediately. Deleting the commit is not enough.

## Reporting a vulnerability

Please **do not** open a public issue. Use GitHub's private reporting instead: on the repository's **Security** tab, choose
**Report a vulnerability**. Only the `main` branch is maintained.
