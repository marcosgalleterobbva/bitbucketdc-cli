# AGENTS.md

This file is the operational reference for agents working on this repo.

## Scope

This CLI is intentionally focused on contributor/reviewer workflows for Bitbucket Data Center / Server.
Admin/system endpoints (permissions, security, mirroring, etc.) are out of scope unless explicitly requested.

## Quickstart

Required environment variables:

- `BITBUCKET_SERVER`: base REST URL ending in `/rest`
- `BITBUCKET_API_TOKEN`: Bitbucket token (PAT or HTTP access token; BBVA commonly uses HTTP access tokens)

Sanity check:

```bash
bbdc doctor
```

Codex runtime caveat:
- For this BBVA environment, agents must never execute `bbdc` in Codex.
- Always provide commands for the user to run locally and continue from user-provided output.

## Core commands (overview)

Top-level:

- `bbdc doctor`
- `bbdc dashboard pull-requests`
- `bbdc account me|recent-repos|ssh-keys|gpg-keys|user|settings`

Pull requests:

- `bbdc pr list|get|create|comment|approve|unapprove|decline|reopen|merge-check|merge|update|watch|unwatch`
- `bbdc pr activities|changes|commits|diff|diff-file|diff-stats|patch|merge-base|commit-message|rebase-check|rebase|delete|for-commit`

Participants:

- `bbdc pr participants list|add|remove|status|search`

Comments:

- `bbdc pr comments add|list|get|update|delete|apply-suggestion|react|unreact`

Blocker comments:

- `bbdc pr blockers list|add|get|update|delete`

Review workflow:

- `bbdc pr review get|complete|discard`

Auto-merge:

- `bbdc pr auto-merge get|set|cancel`

Full options are in `docs/CLI.md`.

## Endpoint mapping

The CLI primarily targets endpoints under `/rest/api/latest`. It also uses non-`/api/latest` endpoints for:

- `comment-likes/latest/...` (comment reactions)
- `git/latest/...` (PR rebase)

These are accessed via `BitbucketClient.request_rest`, which uses the base REST URL without forcing `/api/latest`.

## Input conventions

- Project: `--project` maps to Bitbucket project key.
- Repo: `--repo` maps to repository slug.
- Reviewer identity: uses `{"user": {"name": "<reviewer>"}}` by default. If your instance expects a different
  field (e.g., slug), adjust in `bbdc_cli/__main__.py`.
- Versioned operations: `pr update`, `pr decline`, `pr merge`, `pr reopen`, `pr delete` accept `--version`.
  If omitted, the CLI auto-fetches the current PR version.
- Comment updates/deletes: `--version` is optional; CLI auto-fetches the comment version.

## Output conventions

- `--json` prints raw JSON for list/get operations.
- `account me` is JSON by default and does not accept `--json`.
- Diffs/patches are streamed as raw text (not JSON).
- Tables are used for PR list and participant list; everything else is JSON or raw.

## Adding new commands

Follow these patterns in `bbdc_cli/__main__.py`:

- Use `bb.request` for `/api/latest` endpoints.
- Use `bb.request_rest` for non-`/api/latest` endpoints (e.g., `comment-likes/latest`, `git/latest`).
- Use `bb.paged_get` for endpoints returning `values` and pagination fields.
- For endpoints that require a version field, add `--version` and auto-fetch using `_get_pr_version` or
  `_get_comment_version` when possible.
- Use `_encode_path` for path parameters that represent file paths.

## Common pitfalls

- `BITBUCKET_SERVER` must end in `/rest`.
- Some endpoints return raw text (diff/patch). `_print_raw` handles both JSON and text.
- `comment-likes` and `git/latest` are not under `/api/latest`.
- For comment updates/deletes, server requires the current comment `version`.

## Documentation assets

- `docs/CLI.md`: complete command reference
- `docs/examples.md`: common workflows
- `docs/coverage.md`: OpenAPI/Postman coverage summary and gaps
