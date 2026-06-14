# Security Policy

## API Keys & Secrets

This project requires several third-party API keys (Google Gemini, MiniMax, Runware, Freesound), configured via a local `.env` file.

- **Never commit your `.env` file.** It is excluded via `.gitignore` — only `.env.example` (with empty placeholder values) should be committed.
- **Never paste real API keys into issues, pull requests, commit messages, or logs.**
- If you accidentally commit or expose an API key:
  1. **Rotate it immediately** at the provider (revoke the old key and generate a new one).
  2. Remove it from git history (e.g. with `git filter-repo` or BFG Repo-Cleaner) — simply deleting the file in a new commit is not enough, since the key remains in history.
  3. Update your local `.env` with the new key.

## Reporting a Vulnerability

If you discover a security issue in this project (e.g. unsafe handling of user input, command injection in the FFmpeg pipeline, etc.), please report it privately rather than opening a public issue:

- Open a [GitHub Security Advisory](../../security/advisories/new) for this repository, or
- Contact the maintainer directly.

Please include a description of the issue, steps to reproduce, and any relevant logs. We'll aim to respond as soon as possible.
