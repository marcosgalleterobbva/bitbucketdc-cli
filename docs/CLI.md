# CLI Reference

This is the authoritative command reference derived from `bbdc_cli/__main__.py`.

Codex runtime note: if commands fail in Codex with DNS/network errors (for example `NameResolutionError`), run the same `bbdc` commands in your local terminal and share output back to Codex.

Global:

```
bbdc doctor [--json]
```

Account:

```
bbdc account recent-repos [--limit N] [--max-items N] [--json]

bbdc account ssh-keys [--user <slug>] [--limit N] [--max-items N] [--json]

bbdc account gpg-keys [--user <slug>] [--limit N] [--max-items N] [--json]

bbdc account user [--user-slug <slug>]

bbdc account settings [--user-slug <slug>]

bbdc account me [--user-slug <slug>]
                [--include-profile|--no-include-profile]
                [--include-settings|--no-include-settings]
                [--limit N] [--max-items N]
```

For `account user`, `account settings`, and profile/settings expansion in `account me`, user slug is resolved from:
- `--user-slug`
- `BITBUCKET_USER_SLUG`
- `BITBUCKET_USERNAME`
- `BITBUCKET_USER`

Pull requests:

```
bbdc pr list --project <KEY> --repo <SLUG> [--state OPEN|DECLINED|MERGED|ALL] [--direction INCOMING|OUTGOING]
             [--limit N] [--max-items N] [--json]

bbdc pr get --project <KEY> --repo <SLUG> <pr_id>

bbdc pr create --project <KEY> --repo <SLUG> --from-branch <name> --to-branch <name>
               --title <text> [--description <text>] [--reviewer <user> ...]
               [--draft|--no-draft] [--json]

bbdc pr comment --project <KEY> --repo <SLUG> <pr_id> --text <text>

bbdc pr approve --project <KEY> --repo <SLUG> <pr_id>
bbdc pr unapprove --project <KEY> --repo <SLUG> <pr_id>

bbdc pr decline --project <KEY> --repo <SLUG> <pr_id> [--version N] [--comment <text>] [--json]

bbdc pr reopen --project <KEY> --repo <SLUG> <pr_id> [--version N] [--json]

bbdc pr merge-check --project <KEY> --repo <SLUG> <pr_id>

bbdc pr merge --project <KEY> --repo <SLUG> <pr_id>
              [--version N] [--message <text>] [--strategy <id>]
              [--auto-merge|--no-auto-merge] [--auto-subject <text>] [--json]

bbdc pr update --project <KEY> --repo <SLUG> <pr_id>
               [--version N] [--title <text>] [--description <text>]
               [--reviewer <user> ...] [--draft|--no-draft] [--json]

bbdc pr watch --project <KEY> --repo <SLUG> <pr_id>
bbdc pr unwatch --project <KEY> --repo <SLUG> <pr_id>

bbdc pr activities --project <KEY> --repo <SLUG> <pr_id>
                   [--from-id <id>] [--from-type COMMENT|ACTIVITY]
                   [--limit N] [--max-items N] [--json]

bbdc pr changes --project <KEY> --repo <SLUG> <pr_id>
                [--change-scope ALL|UNREVIEWED|RANGE]
                [--since-id <hash>] [--until-id <hash>]
                [--with-comments|--no-with-comments]
                [--limit N] [--max-items N] [--json]

bbdc pr commits --project <KEY> --repo <SLUG> <pr_id>
                [--with-counts|--no-with-counts]
                [--avatar-size N] [--avatar-scheme http|https]
                [--limit N] [--max-items N] [--json]

bbdc pr diff --project <KEY> --repo <SLUG> <pr_id>
             [--context-lines N] [--whitespace ignore-all]

bbdc pr diff-file --project <KEY> --repo <SLUG> <pr_id> <path>
                  [--since-id <hash>] [--until-id <hash>] [--src-path <oldpath>]
                  [--diff-type <type>] [--context-lines N] [--whitespace ignore-all]
                  [--with-comments|--no-with-comments]
                  [--avatar-size N] [--avatar-scheme http|https]

bbdc pr diff-stats --project <KEY> --repo <SLUG> <pr_id> <path>
                   [--since-id <hash>] [--until-id <hash>] [--src-path <oldpath>]
                   [--whitespace ignore-all]

bbdc pr patch --project <KEY> --repo <SLUG> <pr_id>

bbdc pr merge-base --project <KEY> --repo <SLUG> <pr_id>

bbdc pr commit-message --project <KEY> --repo <SLUG> <pr_id>

bbdc pr rebase-check --project <KEY> --repo <SLUG> <pr_id>

bbdc pr rebase --project <KEY> --repo <SLUG> <pr_id> [--version N] [--json]

bbdc pr delete --project <KEY> --repo <SLUG> <pr_id> [--version N] [--json]

bbdc pr for-commit --project <KEY> --repo <SLUG> <commit_id>
                   [--limit N] [--max-items N] [--json]
```

Batch operations:

Batch commands read a JSON list of objects from `--file` (or `-` for stdin). Common options:

- `--file <path|->` (required)
- `[--project <KEY>] [--repo <SLUG>]` default fields applied to each item
- `[--defaults <json|@file>]` additional default fields (JSON object or `@` file)
- `[--concurrency N]` (default 1)
- `[--continue-on-error|--stop-on-error]` (default continue)
- `[--json]` print raw JSON results

Item fields mirror the corresponding command options using snake_case (e.g., `pr_id`, `from_branch`). Reviewers can be
provided as `reviewers` (list) or `reviewer` (single). Version fields are optional; when omitted, the CLI will fetch
the current version from the server where required. Per-item fields override defaults; command-line defaults override
`--defaults`.

Batch is provided for single-entity actions and simple GETs. Paged list and streaming endpoints (e.g., list/activities/
changes/commits/diff/patch) remain per-invocation.

Batch pull requests:

```
bbdc pr batch get --file <path>
bbdc pr batch create --file <path>
bbdc pr batch comment --file <path>
bbdc pr batch approve --file <path>
bbdc pr batch unapprove --file <path>
bbdc pr batch decline --file <path>
bbdc pr batch reopen --file <path>
bbdc pr batch merge-check --file <path>
bbdc pr batch merge --file <path>
bbdc pr batch update --file <path>
bbdc pr batch watch --file <path>
bbdc pr batch unwatch --file <path>
bbdc pr batch merge-base --file <path>
bbdc pr batch commit-message --file <path>
bbdc pr batch rebase-check --file <path>
bbdc pr batch rebase --file <path>
bbdc pr batch delete --file <path>
```

Participants:

```
bbdc pr participants list --project <KEY> --repo <SLUG> <pr_id>
                         [--limit N] [--max-items N] [--json]

bbdc pr participants add --project <KEY> --repo <SLUG> <pr_id>
                        --user <username> [--role AUTHOR|REVIEWER|PARTICIPANT] [--json]

bbdc pr participants remove --project <KEY> --repo <SLUG> <pr_id> <user_slug>

bbdc pr participants status --project <KEY> --repo <SLUG> <pr_id> <user_slug>
                           --status UNAPPROVED|NEEDS_WORK|APPROVED
                           [--last-reviewed-commit <hash>] [--version N] [--json]

bbdc pr participants search --project <KEY> --repo <SLUG>
                           [--filter <text>] [--role AUTHOR|REVIEWER|PARTICIPANT]
                           [--direction INCOMING|OUTGOING]
                           [--limit N] [--max-items N] [--json]

bbdc pr batch participants add --file <path>
bbdc pr batch participants remove --file <path>
bbdc pr batch participants status --file <path>
```

Comments:

```
bbdc pr comments add --project <KEY> --repo <SLUG> <pr_id> --text <text>

bbdc pr comments list --project <KEY> --repo <SLUG> <pr_id>
                      --path <file_path>
                      [--from-hash <hash>] [--to-hash <hash>]
                      [--diff-types <list>] [--states <list>] [--anchor-state ACTIVE|ORPHANED|ALL]
                      [--limit N] [--max-items N] [--json]

bbdc pr comments get --project <KEY> --repo <SLUG> <pr_id> <comment_id>

bbdc pr comments update --project <KEY> --repo <SLUG> <pr_id> <comment_id>
                        [--text <text>] [--severity NORMAL|BLOCKER] [--state OPEN|RESOLVED]
                        [--version N] [--json]

bbdc pr comments delete --project <KEY> --repo <SLUG> <pr_id> <comment_id>
                        [--version N]

bbdc pr comments apply-suggestion --project <KEY> --repo <SLUG> <pr_id> <comment_id>
                                 --suggestion-index N
                                 [--comment-version N] [--pr-version N]
                                 [--commit-message <text>] [--json]

bbdc pr comments react --project <KEY> --repo <SLUG> <pr_id> <comment_id> --emoticon ":+1:"

bbdc pr comments unreact --project <KEY> --repo <SLUG> <pr_id> <comment_id> --emoticon ":+1:"

bbdc pr batch comments add --file <path>
bbdc pr batch comments get --file <path>
bbdc pr batch comments update --file <path>
bbdc pr batch comments delete --file <path>
bbdc pr batch comments apply-suggestion --file <path>
bbdc pr batch comments react --file <path>
bbdc pr batch comments unreact --file <path>
```

Blocker comments:

```
bbdc pr blockers list --project <KEY> --repo <SLUG> <pr_id>
                      [--states <list>] [--count]
                      [--limit N] [--max-items N] [--json]

bbdc pr blockers add --project <KEY> --repo <SLUG> <pr_id> --text <text> [--json]

bbdc pr blockers get --project <KEY> --repo <SLUG> <pr_id> <comment_id>

bbdc pr blockers update --project <KEY> --repo <SLUG> <pr_id> <comment_id>
                        [--text <text>] [--severity NORMAL|BLOCKER] [--state OPEN|RESOLVED]
                        [--version N] [--json]

bbdc pr blockers delete --project <KEY> --repo <SLUG> <pr_id> <comment_id>
                        [--version N]

bbdc pr batch blockers add --file <path>
bbdc pr batch blockers get --file <path>
bbdc pr batch blockers update --file <path>
bbdc pr batch blockers delete --file <path>
```

Review:

```
bbdc pr review get --project <KEY> --repo <SLUG> <pr_id>

bbdc pr review complete --project <KEY> --repo <SLUG> <pr_id>
                        [--comment <text>] [--last-reviewed-commit <hash>]
                        [--status UNAPPROVED|NEEDS_WORK|APPROVED] [--json]

bbdc pr review discard --project <KEY> --repo <SLUG> <pr_id>

bbdc pr batch review get --file <path>
bbdc pr batch review complete --file <path>
bbdc pr batch review discard --file <path>
```

Auto-merge:

```
bbdc pr auto-merge get --project <KEY> --repo <SLUG> <pr_id>

bbdc pr auto-merge set --project <KEY> --repo <SLUG> <pr_id>

bbdc pr auto-merge cancel --project <KEY> --repo <SLUG> <pr_id>

bbdc pr batch auto-merge get --file <path>
bbdc pr batch auto-merge set --file <path>
bbdc pr batch auto-merge cancel --file <path>
```
