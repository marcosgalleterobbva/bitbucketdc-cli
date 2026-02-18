# Examples

These examples use a fictional project `GL_KAIF_APP-ID-2866825_DSG` and repo `mercury-viz`.

## Inspect your authenticated account

```bash
# Consolidated account snapshot (recent repos, SSH keys, GPG keys)
bbdc account me

# Show partial/error structure explicitly
bbdc account me --json

# Include profile/settings when your slug is known
bbdc account me --user-slug alice --include-settings

# Individual account endpoints
bbdc account recent-repos
bbdc account ssh-keys
bbdc account gpg-keys
bbdc account user --user-slug alice
bbdc account settings --user-slug alice
```

## Review a PR end-to-end

```bash
# List open PRs
bbdc pr list -p GL_KAIF_APP-ID-2866825_DSG -r mercury-viz

# Get a specific PR
bbdc pr get -p GL_KAIF_APP-ID-2866825_DSG -r mercury-viz 123

# Add a comment
bbdc pr comments add -p GL_KAIF_APP-ID-2866825_DSG -r mercury-viz 123 --text "LGTM"

# Approve
bbdc pr approve -p GL_KAIF_APP-ID-2866825_DSG -r mercury-viz 123

# Complete review
bbdc pr review complete -p GL_KAIF_APP-ID-2866825_DSG -r mercury-viz 123 \
  --comment "Reviewed" --status APPROVED
```

## Add reviewers to an existing PR

```bash
# Add one reviewer
bbdc pr participants add -p GL_KAIF_APP-ID-2866825_DSG -r mercury-viz 123 \
  --user alice --role REVIEWER

# Replace reviewers list
bbdc pr update -p GL_KAIF_APP-ID-2866825_DSG -r mercury-viz 123 \
  --reviewer alice --reviewer bob
```

## Apply a code suggestion from a comment

```bash
# Fetch comment details (to find suggestion index and version)
bbdc pr comments get -p GL_KAIF_APP-ID-2866825_DSG -r mercury-viz 123 456

# Apply suggestion index 0
bbdc pr comments apply-suggestion -p GL_KAIF_APP-ID-2866825_DSG -r mercury-viz 123 456 \
  --suggestion-index 0
```

## Work with blocker comments (tasks)

```bash
# Add a blocker comment
bbdc pr blockers add -p GL_KAIF_APP-ID-2866825_DSG -r mercury-viz 123 --text "Please fix tests"

# List blocker comments
bbdc pr blockers list -p GL_KAIF_APP-ID-2866825_DSG -r mercury-viz 123

# Resolve a blocker comment (requires comment version; auto-fetched if omitted)
bbdc pr blockers update -p GL_KAIF_APP-ID-2866825_DSG -r mercury-viz 123 789 --state RESOLVED
```

## Inspect diffs and changes

```bash
# Raw PR diff
bbdc pr diff -p GL_KAIF_APP-ID-2866825_DSG -r mercury-viz 123

# Diff for a file
bbdc pr diff-file -p GL_KAIF_APP-ID-2866825_DSG -r mercury-viz 123 src/main.py

# Diff stats summary for a file
bbdc pr diff-stats -p GL_KAIF_APP-ID-2866825_DSG -r mercury-viz 123 src/main.py

# Changes with comment counts
bbdc pr changes -p GL_KAIF_APP-ID-2866825_DSG -r mercury-viz 123 --with-comments
```

## Rebase and merge

```bash
# Check whether rebase is possible
bbdc pr rebase-check -p GL_KAIF_APP-ID-2866825_DSG -r mercury-viz 123

# Rebase (auto-fetches PR version)
bbdc pr rebase -p GL_KAIF_APP-ID-2866825_DSG -r mercury-viz 123

# Merge with a message
bbdc pr merge -p GL_KAIF_APP-ID-2866825_DSG -r mercury-viz 123 --message "LGTM"
```

## Find PRs for a commit

```bash
bbdc pr for-commit -p GL_KAIF_APP-ID-2866825_DSG -r mercury-viz 8d51122def56
```

## Batch create PRs

Example batch file (`batch-prs.json`):

```json
[
  {
    "from_branch": "feature/one",
    "to_branch": "develop",
    "title": "Add feature one",
    "description": "Implements feature one",
    "reviewers": ["alice", "bob"]
  },
  {
    "from_branch": "feature/two",
    "to_branch": "develop",
    "title": "Add feature two",
    "draft": true
  }
]
```

Run (defaults apply to each item):

```bash
bbdc pr batch create -p GL_KAIF_APP-ID-2866825_DSG -r mercury-viz -f batch-prs.json
```

## Batch approve PRs

```json
[
  {"pr_id": 123},
  {"pr_id": 456},
  {"pr_id": 789}
]
```

```bash
bbdc pr batch approve -p GL_KAIF_APP-ID-2866825_DSG -r mercury-viz -f approve.json
```
