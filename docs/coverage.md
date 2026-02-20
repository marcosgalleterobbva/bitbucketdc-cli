# Coverage

This document summarizes CLI coverage against the OpenAPI and Postman references in `resources/`.

## Sources

- OpenAPI: `resources/10.0.swagger.v3.json`
- Postman: `resources/bitbucketserver.1000.postman.txt`

Both sources describe 529 API operations for Bitbucket Data Center 10.0.

## Scope

This CLI is intentionally focused on contributor/reviewer workflows. Admin/system endpoints (permissions,
security, mirroring, etc.) are out of scope unless explicitly requested.

## Implemented (contributor/reviewer)

Below are the endpoints implemented in `bbdc_cli/__main__.py` that map to contributor/reviewer workflows.

### Pull requests (core)

- `GET /api/latest/projects/{projectKey}/repos/{repositorySlug}/pull-requests`
- `GET /api/latest/projects/{projectKey}/repos/{repositorySlug}/pull-requests/{pullRequestId}`
- `POST /api/latest/projects/{projectKey}/repos/{repositorySlug}/pull-requests`
- `PUT /api/latest/projects/{projectKey}/repos/{repositorySlug}/pull-requests/{pullRequestId}`
- `DELETE /api/latest/projects/{projectKey}/repos/{repositorySlug}/pull-requests/{pullRequestId}`

### Pull request lifecycle

- `POST /api/latest/projects/{projectKey}/repos/{repositorySlug}/pull-requests/{pullRequestId}/approve`
- `DELETE /api/latest/projects/{projectKey}/repos/{repositorySlug}/pull-requests/{pullRequestId}/approve`
- `POST /api/latest/projects/{projectKey}/repos/{repositorySlug}/pull-requests/{pullRequestId}/decline`
- `POST /api/latest/projects/{projectKey}/repos/{repositorySlug}/pull-requests/{pullRequestId}/merge`
- `GET /api/latest/projects/{projectKey}/repos/{repositorySlug}/pull-requests/{pullRequestId}/merge`
- `POST /api/latest/projects/{projectKey}/repos/{repositorySlug}/pull-requests/{pullRequestId}/reopen`

### Participants / reviewers

- `GET /api/latest/projects/{projectKey}/repos/{repositorySlug}/pull-requests/{pullRequestId}/participants`
- `POST /api/latest/projects/{projectKey}/repos/{repositorySlug}/pull-requests/{pullRequestId}/participants`
- `DELETE /api/latest/projects/{projectKey}/repos/{repositorySlug}/pull-requests/{pullRequestId}/participants/{userSlug}`
- `PUT /api/latest/projects/{projectKey}/repos/{repositorySlug}/pull-requests/{pullRequestId}/participants/{userSlug}`
- `GET /api/latest/projects/{projectKey}/repos/{repositorySlug}/participants`

### Comments and suggestions

- `POST /api/latest/projects/{projectKey}/repos/{repositorySlug}/pull-requests/{pullRequestId}/comments`
- `GET /api/latest/projects/{projectKey}/repos/{repositorySlug}/pull-requests/{pullRequestId}/comments`
- `GET /api/latest/projects/{projectKey}/repos/{repositorySlug}/pull-requests/{pullRequestId}/comments/{commentId}`
- `PUT /api/latest/projects/{projectKey}/repos/{repositorySlug}/pull-requests/{pullRequestId}/comments/{commentId}`
- `DELETE /api/latest/projects/{projectKey}/repos/{repositorySlug}/pull-requests/{pullRequestId}/comments/{commentId}`
- `POST /api/latest/projects/{projectKey}/repos/{repositorySlug}/pull-requests/{pullRequestId}/comments/{commentId}/apply-suggestion`

### Comment reactions (non-`/api/latest`)

- `PUT /rest/comment-likes/latest/projects/{projectKey}/repos/{repositorySlug}/pull-requests/{pullRequestId}/comments/{commentId}/reactions/{emoticon}`
- `DELETE /rest/comment-likes/latest/projects/{projectKey}/repos/{repositorySlug}/pull-requests/{pullRequestId}/comments/{commentId}/reactions/{emoticon}`

### Blocker comments (tasks)

- `GET /api/latest/projects/{projectKey}/repos/{repositorySlug}/pull-requests/{pullRequestId}/blocker-comments`
- `POST /api/latest/projects/{projectKey}/repos/{repositorySlug}/pull-requests/{pullRequestId}/blocker-comments`
- `GET /api/latest/projects/{projectKey}/repos/{repositorySlug}/pull-requests/{pullRequestId}/blocker-comments/{commentId}`
- `PUT /api/latest/projects/{projectKey}/repos/{repositorySlug}/pull-requests/{pullRequestId}/blocker-comments/{commentId}`
- `DELETE /api/latest/projects/{projectKey}/repos/{repositorySlug}/pull-requests/{pullRequestId}/blocker-comments/{commentId}`

### Review workflow

- `GET /api/latest/projects/{projectKey}/repos/{repositorySlug}/pull-requests/{pullRequestId}/review`
- `PUT /api/latest/projects/{projectKey}/repos/{repositorySlug}/pull-requests/{pullRequestId}/review`
- `DELETE /api/latest/projects/{projectKey}/repos/{repositorySlug}/pull-requests/{pullRequestId}/review`

### Auto-merge

- `GET /api/latest/projects/{projectKey}/repos/{repositorySlug}/pull-requests/{pullRequestId}/auto-merge`
- `POST /api/latest/projects/{projectKey}/repos/{repositorySlug}/pull-requests/{pullRequestId}/auto-merge`
- `DELETE /api/latest/projects/{projectKey}/repos/{repositorySlug}/pull-requests/{pullRequestId}/auto-merge`

### Activity, changes, commits, diffs

- `GET /api/latest/projects/{projectKey}/repos/{repositorySlug}/pull-requests/{pullRequestId}/activities`
- `GET /api/latest/projects/{projectKey}/repos/{repositorySlug}/pull-requests/{pullRequestId}/changes`
- `GET /api/latest/projects/{projectKey}/repos/{repositorySlug}/pull-requests/{pullRequestId}/commits`
- `GET /api/latest/projects/{projectKey}/repos/{repositorySlug}/pull-requests/{pullRequestId}.diff`
- `GET /api/latest/projects/{projectKey}/repos/{repositorySlug}/pull-requests/{pullRequestId}.patch`
- `GET /api/latest/projects/{projectKey}/repos/{repositorySlug}/pull-requests/{pullRequestId}/diff/{path}`
- `GET /api/latest/projects/{projectKey}/repos/{repositorySlug}/pull-requests/{pullRequestId}/diff-stats-summary/{path}`
- `GET /api/latest/projects/{projectKey}/repos/{repositorySlug}/pull-requests/{pullRequestId}/merge-base`
- `GET /api/latest/projects/{projectKey}/repos/{repositorySlug}/pull-requests/{pullRequestId}/commit-message-suggestion`

### Rebase (non-`/api/latest`)

- `GET /rest/git/latest/projects/{projectKey}/repos/{repositorySlug}/pull-requests/{pullRequestId}/rebase`
- `POST /rest/git/latest/projects/{projectKey}/repos/{repositorySlug}/pull-requests/{pullRequestId}/rebase`

### PRs for a commit

- `GET /api/latest/projects/{projectKey}/repos/{repositorySlug}/commits/{commitId}/pull-requests`

### Dashboard

- `GET /api/latest/dashboard/pull-requests`

## Not implemented (contributor/reviewer)

These are contributor-focused endpoints that are *not* implemented yet:

- Dashboard suggestions: `GET /api/latest/dashboard/pull-request-suggestions`
- Repository browsing:
  - `GET /api/latest/projects/{projectKey}/repos/{repositorySlug}/browse` and `.../browse/{path}`
  - `GET /api/latest/projects/{projectKey}/repos/{repositorySlug}/raw/{path}`
  - `GET /api/latest/projects/{projectKey}/repos/{repositorySlug}/files` and `.../files/{path}`
- Commit inspection outside PR context:
  - `GET /api/latest/projects/{projectKey}/repos/{repositorySlug}/commits`
  - `GET /api/latest/projects/{projectKey}/repos/{repositorySlug}/commits/{commitId}`
  - `GET /api/latest/projects/{projectKey}/repos/{repositorySlug}/commits/{commitId}/diff{...}`
  - `GET /api/latest/projects/{projectKey}/repos/{repositorySlug}/commits/{commitId}/changes`
- Repo-level diffs and compare endpoints:
  - `GET /api/latest/projects/{projectKey}/repos/{repositorySlug}/diff{...}`
  - `GET /api/latest/projects/{projectKey}/repos/{repositorySlug}/compare/commits`
  - `GET /api/latest/projects/{projectKey}/repos/{repositorySlug}/compare/changes`
- Repository tags/branches (read):
  - `GET /api/latest/projects/{projectKey}/repos/{repositorySlug}/branches`
  - `GET /api/latest/projects/{projectKey}/repos/{repositorySlug}/tags`
- Search endpoints (indexing/status in `indexing/latest`)

If you want these added, open a task and specify which ones are highest priority.

## Out of scope (admin/system)

Examples of intentionally excluded areas:

- Permissions, security, system maintenance
- Mirroring and synchronization
- Global authentication administration
- Project and repo administration (hooks, webhooks, restrictions)
