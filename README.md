# bbdc-cli

A small, practical Typer CLI for Bitbucket Data Center / Server REST API.

It reads credentials from environment variables and provides high-signal PR workflows (list, create, comment,
approve, merge, update metadata, manage reviewers/participants, review completion, diffs, etc.) without needing a
full SDK.

## Requirements

- Python 3.9+
- `pipx` recommended for isolated install

## Install

From PyPI:

```bash
pipx install bbdc-cli
# or
pip install bbdc-cli
```

From source (repo root with `pyproject.toml`):

```bash
pipx install .

# If you are iterating locally:
pipx install -e .

# Reinstall after changes (non-editable install):
pipx reinstall bbdc-cli

# Uninstall:
pipx uninstall bbdc-cli
```

## Configuration

The CLI uses two environment variables:

- `BITBUCKET_SERVER`: base REST URL ending in `/rest`
- `BITBUCKET_API_TOKEN`: Bitbucket personal access token (PAT)

Example (BBVA-style context path):

```
https://bitbucket.globaldevtools.bbva.com/bitbucket/rest
```

Set them:

```bash
export BITBUCKET_SERVER="https://bitbucket.globaldevtools.bbva.com/bitbucket/rest"
export BITBUCKET_API_TOKEN="YOUR_TOKEN"
```

## Quick check

```bash
bbdc doctor
# machine-readable output
bbdc doctor --json
```

If this succeeds, your base URL + token are working.

Optional (for account profile/settings lookups):

- `BITBUCKET_USER_SLUG`: your Bitbucket user slug

## Common commands

Show help:

```bash
bbdc --help
bbdc account --help
bbdc pr --help
```

Get information about your authenticated account:

```bash
# consolidated snapshot (recent repos + SSH keys + GPG keys)
bbdc account me

# include user profile and settings when your slug is known
bbdc account me --user-slug your.user --include-settings

# raw account endpoint calls
bbdc account recent-repos
bbdc account ssh-keys
bbdc account gpg-keys
bbdc account user --user-slug your.user
bbdc account settings --user-slug your.user
```

List pull requests:

```bash
bbdc pr list --project GL_KAIF_APP-ID-2866825_DSG --repo mercury-viz
```

Get a pull request:

```bash
bbdc pr get -p GL_KAIF_APP-ID-2866825_DSG -r mercury-viz 123
```

Create a pull request:

```bash
bbdc pr create \
  --project GL_KAIF_APP-ID-2866825_DSG \
  --repo mercury-viz \
  --from-branch feature/my-branch \
  --to-branch develop \
  --title "Add viz panel" \
  --description "Implements X"
```

Add reviewers (repeat `--reviewer`):

```bash
bbdc pr create \
  -p GL_KAIF_APP-ID-2866825_DSG \
  -r mercury-viz \
  --from-branch feature/my-branch \
  --to-branch develop \
  --title "Add viz panel" \
  --description "Implements X" \
  --reviewer some.username \
  --reviewer other.username
```

Approve, decline, merge:

```bash
bbdc pr approve -p GL_KAIF_APP-ID-2866825_DSG -r mercury-viz 123
bbdc pr decline -p GL_KAIF_APP-ID-2866825_DSG -r mercury-viz 123 --comment "Not proceeding"
bbdc pr merge -p GL_KAIF_APP-ID-2866825_DSG -r mercury-viz 123 --message "LGTM"
```

Update metadata:

```bash
bbdc pr update -p GL_KAIF_APP-ID-2866825_DSG -r mercury-viz 123 \
  --title "New title" \
  --description "Updated description" \
  --reviewer some.username
```

Participants / reviewers:

```bash
bbdc pr participants list -p GL_KAIF_APP-ID-2866825_DSG -r mercury-viz 123
bbdc pr participants add -p GL_KAIF_APP-ID-2866825_DSG -r mercury-viz 123 --user alice --role REVIEWER
bbdc pr participants status -p GL_KAIF_APP-ID-2866825_DSG -r mercury-viz 123 alice --status APPROVED
```

Review completion and comments:

```bash
bbdc pr review complete -p GL_KAIF_APP-ID-2866825_DSG -r mercury-viz 123 --comment "Looks good" --status APPROVED
bbdc pr comments add -p GL_KAIF_APP-ID-2866825_DSG -r mercury-viz 123 --text "LGTM"
```

## Batch operations

Batch commands live under `bbdc pr batch ...` and read a JSON list of items from `--file` (or `-` for stdin). You can
provide `--project` and `--repo` as defaults for each item.

Example batch approvals (`approve.json`):

```json
[
  {"pr_id": 123},
  {"pr_id": 456}
]
```

```bash
bbdc pr batch approve -p GL_KAIF_APP-ID-2866825_DSG -r mercury-viz -f approve.json
```

Diffs and commits:

```bash
bbdc pr commits -p GL_KAIF_APP-ID-2866825_DSG -r mercury-viz 123
bbdc pr diff -p GL_KAIF_APP-ID-2866825_DSG -r mercury-viz 123
bbdc pr diff-file -p GL_KAIF_APP-ID-2866825_DSG -r mercury-viz 123 src/main.py
```

See the full command reference in `docs/CLI.md` and usage examples in `docs/examples.md`.

## Codex integration

If teammates will use this through Codex with natural language:

1. Distribute this CLI through PyPI (`bbdc-cli`).
2. Distribute the Codex skill separately (for example, git repo cloned into `$CODEX_HOME/skills/bbdc-cli`).
3. Keep the skill's command inventory synced with this repo's `bbdc_cli/__main__.py`.

Recommended split of responsibilities:
- This repo: command behavior, API semantics, package distribution.
- Skill repo: natural-language intent mapping, execution policy, Codex-specific prompting.

This separation is the correct approach and avoids coupling Codex behavior to package release timing.

## Troubleshooting

`BITBUCKET_SERVER` must end with `/rest`.

Use the REST base, not the UI URL. For instances hosted under `/bitbucket`, the REST base is often:

- UI: `https://host/bitbucket/...`
- REST: `https://host/bitbucket/rest`

Unauthorized / 401 / 403:

- Token missing or incorrect
- Token lacks required permissions for that project/repo
- Your Bitbucket instance may require a different auth scheme (rare if PAT is enabled)

404 Not Found:

Usually one of:

- Wrong `BITBUCKET_SERVER` base path (`/rest` vs `/bitbucket/rest`)
- Wrong `--project` key or `--repo` slug
- PR id does not exist in that repo

## Development

Run without installing:

```bash
python -m bbdc_cli --help
python -m bbdc_cli doctor
```

## License

Mercury - BBVA
