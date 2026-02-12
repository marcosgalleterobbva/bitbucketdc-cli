#!/usr/bin/env python3
"""
bbdc.py — A tiny Typer CLI for Bitbucket Data Center (Server/DC) REST API.

Auth & base URL are taken from environment variables:
  - BITBUCKET_SERVER     e.g. https://bitbucket.example.com/bitbucket/rest   (must end with /rest)
  - BITBUCKET_API_TOKEN  Personal Access Token (PAT)

Install deps:
  python -m pip install typer[all] requests

Examples:
  export BITBUCKET_SERVER="https://bitbucket.globaldevtools.bbva.com/bitbucket/rest"
  export BITBUCKET_API_TOKEN="***"

  # List open PRs
  python bbdc.py pr list --project GL_KAIF_APP-ID-2866825_DSG --repo mercury-viz

  # Create a PR
  python bbdc.py pr create --project GL_KAIF_APP-ID-2866825_DSG --repo mercury-viz \
    --from-branch feature/my-branch --to-branch develop \
    --title "Add viz panel" --description "Implements X" \
    --reviewer some.username --reviewer other.username
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from urllib.parse import quote
from typing import Any, Dict, Iterable, List, Optional

import requests
import typer

app = typer.Typer(no_args_is_help=True, add_completion=False)
pr_app = typer.Typer(no_args_is_help=True, add_completion=False)
participants_app = typer.Typer(no_args_is_help=True, add_completion=False)
comments_app = typer.Typer(no_args_is_help=True, add_completion=False)
blockers_app = typer.Typer(no_args_is_help=True, add_completion=False)
review_app = typer.Typer(no_args_is_help=True, add_completion=False)
auto_merge_app = typer.Typer(no_args_is_help=True, add_completion=False)
app.add_typer(pr_app, name="pr", help="Pull request operations")
pr_app.add_typer(participants_app, name="participants", help="PR participants and reviewers")
pr_app.add_typer(comments_app, name="comments", help="PR comments")
pr_app.add_typer(blockers_app, name="blockers", help="PR blocker comments")
pr_app.add_typer(review_app, name="review", help="PR review workflow")
pr_app.add_typer(auto_merge_app, name="auto-merge", help="PR auto-merge")


class BBError(RuntimeError):
    pass


def _env(name: str) -> str:
    v = os.getenv(name, "").strip()
    if not v:
        raise BBError(f"Missing environment variable {name}.")
    return v


def _norm_base(server: str) -> str:
    server = server.rstrip("/")
    if not server.endswith("/rest"):
        raise BBError(
            "BITBUCKET_SERVER must end with '/rest' (example: https://host/bitbucket/rest). "
            f"Got: {server}"
        )
    return server


@dataclass(frozen=True)
class BitbucketClient:
    base_rest: str
    token: str
    timeout_s: int = 30

    @property
    def api(self) -> str:
        # Postman collection uses api/latest
        return f"{self.base_rest}/api/latest"

    def _headers(self, content_type: Optional[str] = None) -> Dict[str, str]:
        h = {
            "Authorization": f"Bearer {self.token}",
            "Accept": "application/json;charset=UTF-8",
        }
        if content_type:
            h["Content-Type"] = content_type
        return h

    def request(
        self,
        method: str,
        path: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        json_body: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        url = f"{self.api}/{path.lstrip('/')}"
        try:
            resp = requests.request(
                method=method.upper(),
                url=url,
                headers=self._headers("application/json" if json_body is not None else None),
                params=params,
                json=json_body,
                timeout=self.timeout_s,
            )
        except requests.RequestException as e:
            raise BBError(f"Request failed: {e}") from e

        if resp.status_code >= 400:
            # Best-effort error extraction
            detail = ""
            try:
                j = resp.json()
                if isinstance(j, dict):
                    if "errors" in j and isinstance(j["errors"], list) and j["errors"]:
                        # Bitbucket often returns: {"errors":[{"message": "..."}]}
                        msg = j["errors"][0].get("message")
                        if msg:
                            detail = msg
                    elif "message" in j and isinstance(j["message"], str):
                        detail = j["message"]
            except Exception:
                pass
            raise BBError(f"HTTP {resp.status_code} for {method} {url}" + (f": {detail}" if detail else ""))

        if not resp.content:
            return {}
        # Some endpoints may return plain text; keep it robust
        ctype = resp.headers.get("content-type", "")
        if "application/json" in ctype:
            return resp.json()
        return {"raw": resp.text}

    def request_rest(
        self,
        method: str,
        path: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        json_body: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        url = f"{self.base_rest}/{path.lstrip('/')}"
        try:
            resp = requests.request(
                method=method.upper(),
                url=url,
                headers=self._headers("application/json" if json_body is not None else None),
                params=params,
                json=json_body,
                timeout=self.timeout_s,
            )
        except requests.RequestException as e:
            raise BBError(f"Request failed: {e}") from e

        if resp.status_code >= 400:
            # Best-effort error extraction
            detail = ""
            try:
                j = resp.json()
                if isinstance(j, dict):
                    if "errors" in j and isinstance(j["errors"], list) and j["errors"]:
                        # Bitbucket often returns: {"errors":[{"message": "..."}]}
                        msg = j["errors"][0].get("message")
                        if msg:
                            detail = msg
                    elif "message" in j and isinstance(j["message"], str):
                        detail = j["message"]
            except Exception:
                pass
            raise BBError(f"HTTP {resp.status_code} for {method} {url}" + (f": {detail}" if detail else ""))

        if not resp.content:
            return {}
        # Some endpoints may return plain text; keep it robust
        ctype = resp.headers.get("content-type", "")
        if "application/json" in ctype:
            return resp.json()
        return {"raw": resp.text}


    def paged_get(
        self,
        path: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        limit: int = 50,
        max_items: int = 200,
    ) -> List[Dict[str, Any]]:
        """Fetch Bitbucket paged results (values/isLastPage/nextPageStart)."""
        out: List[Dict[str, Any]] = []
        start = 0
        params = dict(params or {})
        while True:
            params.update({"start": start, "limit": limit})
            page = self.request("GET", path, params=params)
            values = page.get("values", [])
            if isinstance(values, list):
                out.extend(values)
            if len(out) >= max_items:
                return out[:max_items]
            if page.get("isLastPage", True):
                return out
            start = page.get("nextPageStart")
            if start is None:
                return out


def client() -> BitbucketClient:
    return BitbucketClient(
        base_rest=_norm_base(_env("BITBUCKET_SERVER")),
        token=_env("BITBUCKET_API_TOKEN"),
    )


def _print_json(data: Any) -> None:
    typer.echo(json.dumps(data, indent=2, ensure_ascii=False))


def _print_prs(prs: Iterable[Dict[str, Any]]) -> None:
    # Lightweight table without extra deps.
    rows = []
    for pr in prs:
        pr_id = pr.get("id", "")
        title = (pr.get("title") or "").replace("\n", " ")
        state = pr.get("state", "")
        from_ref = pr.get("fromRef", {}).get("displayId") or pr.get("fromRef", {}).get("id", "")
        to_ref = pr.get("toRef", {}).get("displayId") or pr.get("toRef", {}).get("id", "")
        author = pr.get("author", {}).get("user", {}).get("displayName") or pr.get("author", {}).get("user", {}).get("name", "")
        rows.append((str(pr_id), state, author, f"{from_ref} -> {to_ref}", title))

    if not rows:
        typer.echo("No pull requests.")
        return

    headers = ("ID", "STATE", "AUTHOR", "REFS", "TITLE")
    widths = [len(h) for h in headers]
    for r in rows:
        for i, cell in enumerate(r):
            widths[i] = max(widths[i], len(cell))

    def fmt_row(r):
        return "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(r))

    typer.echo(fmt_row(headers))
    typer.echo("  ".join("-" * w for w in widths))
    for r in rows:
        typer.echo(fmt_row(r))


def _print_participants(items: Iterable[Dict[str, Any]]) -> None:
    rows = []
    for p in items:
        user = p.get("user") or {}
        name = user.get("displayName") or user.get("name") or user.get("slug") or ""
        role = p.get("role") or ""
        status = p.get("status") or ""
        approved = p.get("approved")
        approved_str = "" if approved is None else str(bool(approved))
        rows.append((name, role, approved_str, status))

    if not rows:
        typer.echo("No participants.")
        return

    headers = ("USER", "ROLE", "APPROVED", "STATUS")
    widths = [len(h) for h in headers]
    for r in rows:
        for i, cell in enumerate(r):
            widths[i] = max(widths[i], len(cell))

    def fmt_row(r):
        return "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(r))

    typer.echo(fmt_row(headers))
    typer.echo("  ".join("-" * w for w in widths))
    for r in rows:
        typer.echo(fmt_row(r))


ROLE_CHOICES = {"AUTHOR", "REVIEWER", "PARTICIPANT"}
STATUS_CHOICES = {"UNAPPROVED", "NEEDS_WORK", "APPROVED"}
COMMENT_STATE_CHOICES = {"OPEN", "RESOLVED"}
COMMENT_SEVERITY_CHOICES = {"NORMAL", "BLOCKER"}


def _norm_choice(value: str, allowed: set[str], name: str) -> str:
    v = value.strip().upper()
    if v not in allowed:
        opts = ", ".join(sorted(allowed))
        raise BBError(f"Invalid {name} '{value}'. Allowed: {opts}.")
    return v


def _get_pr_version(bb: BitbucketClient, project: str, repo: str, pr_id: int) -> int:
    path = f"projects/{project}/repos/{repo}/pull-requests/{pr_id}"
    pr = bb.request("GET", path)
    version = pr.get("version")
    if version is None:
        raise BBError("Could not determine PR version. Pass --version explicitly.")
    try:
        return int(version)
    except (TypeError, ValueError):
        raise BBError(f"Invalid PR version returned: {version}")


def _print_raw(resp: Any) -> None:
    if isinstance(resp, dict) and "raw" in resp:
        typer.echo(resp["raw"])
    else:
        _print_json(resp)


def _encode_path(path: str) -> str:
    return quote(path, safe="/")


def _get_comment_version(
    bb: BitbucketClient,
    project: str,
    repo: str,
    pr_id: int,
    comment_id: int,
    *,
    blocker: bool = False,
) -> int:
    endpoint = "blocker-comments" if blocker else "comments"
    path = f"projects/{project}/repos/{repo}/pull-requests/{pr_id}/{endpoint}/{comment_id}"
    comment = bb.request("GET", path)
    version = comment.get("version")
    if version is None:
        raise BBError("Could not determine comment version. Pass --version explicitly.")
    try:
        return int(version)
    except (TypeError, ValueError):
        raise BBError(f"Invalid comment version returned: {version}")


@pr_app.command("list")
def pr_list(
    project: str = typer.Option(..., "--project", "-p", help="Project key, e.g. GL_KAIF_APP-ID-2866825_DSG"),
    repo: str = typer.Option(..., "--repo", "-r", help="Repository slug, e.g. mercury-viz"),
    state: str = typer.Option("OPEN", help="OPEN, DECLINED, MERGED, or ALL (Bitbucket semantics)"),
    direction: str = typer.Option("INCOMING", help="INCOMING or OUTGOING"),
    limit: int = typer.Option(50, help="Page size"),
    max_items: int = typer.Option(200, help="Max items to fetch across pages"),
    json_out: bool = typer.Option(False, "--json", help="Print raw JSON instead of a table"),
):
    """
    List pull requests for a repository.

    Corresponds to Postman: Pull Requests -> Get pull requests for repository (GET)
    """
    bb = client()
    path = f"projects/{project}/repos/{repo}/pull-requests"
    prs = bb.paged_get(
        path,
        params={"state": state, "direction": direction},
        limit=limit,
        max_items=max_items,
    )
    if json_out:
        _print_json(prs)
    else:
        _print_prs(prs)


@pr_app.command("get")
def pr_get(
    project: str = typer.Option(..., "--project", "-p"),
    repo: str = typer.Option(..., "--repo", "-r"),
    pr_id: int = typer.Argument(..., help="Pull request numeric ID"),
):
    """Get a single pull request as JSON."""
    bb = client()
    path = f"projects/{project}/repos/{repo}/pull-requests/{pr_id}"
    pr = bb.request("GET", path)
    _print_json(pr)


@pr_app.command("create")
def pr_create(
    project: str = typer.Option(..., "--project", "-p"),
    repo: str = typer.Option(..., "--repo", "-r"),
    from_branch: str = typer.Option(..., "--from-branch", help="Source branch name (without refs/heads/)"),
    to_branch: str = typer.Option(..., "--to-branch", help="Target branch name (without refs/heads/)"),
    title: str = typer.Option(..., "--title"),
    description: str = typer.Option("", "--description"),
    reviewer: List[str] = typer.Option(
        [],
        "--reviewer",
        help="Reviewer username (repeatable). Exact field may vary by instance; this uses user.name.",
    ),
    draft: Optional[bool] = typer.Option(None, "--draft/--no-draft", help="If supported, set PR draft status"),
    json_out: bool = typer.Option(False, "--json", help="Print raw JSON response"),
):
    """
    Create a pull request.

    Corresponds to Postman: Pull Requests -> Create pull request (POST)
    """
    bb = client()

    body: Dict[str, Any] = {
        "title": title,
        "description": description,
        "fromRef": {
            "id": f"refs/heads/{from_branch}",
            "repository": {"slug": repo, "project": {"key": project}},
        },
        "toRef": {
            "id": f"refs/heads/{to_branch}",
            "repository": {"slug": repo, "project": {"key": project}},
        },
    }

    if reviewer:
        body["reviewers"] = [{"user": {"name": r}} for r in reviewer]

    if draft is not None:
        # Bitbucket DC supports draft PRs in newer versions; if unsupported, server will 400.
        body["draft"] = bool(draft)

    path = f"projects/{project}/repos/{repo}/pull-requests"
    created = bb.request("POST", path, json_body=body)

    if json_out:
        _print_json(created)
        return

    pr_id = created.get("id", "?")
    links = created.get("links", {}).get("self", [])
    url = links[0].get("href") if isinstance(links, list) and links else None
    typer.echo(f"Created PR #{pr_id}" + (f": {url}" if url else ""))


@pr_app.command("comment")
def pr_comment(
    project: str = typer.Option(..., "--project", "-p"),
    repo: str = typer.Option(..., "--repo", "-r"),
    pr_id: int = typer.Argument(..., help="Pull request numeric ID"),
    text: str = typer.Option(..., "--text", "-t", help="Comment text"),
):
    """Add a comment to a pull request."""
    bb = client()
    path = f"projects/{project}/repos/{repo}/pull-requests/{pr_id}/comments"
    body = {"text": text}
    resp = bb.request("POST", path, json_body=body)
    comment_id = resp.get("id", "?")
    typer.echo(f"Added comment {comment_id} to PR #{pr_id}")


@pr_app.command("approve")
def pr_approve(
    project: str = typer.Option(..., "--project", "-p"),
    repo: str = typer.Option(..., "--repo", "-r"),
    pr_id: int = typer.Argument(..., help="Pull request numeric ID"),
):
    """Approve a pull request."""
    bb = client()
    path = f"projects/{project}/repos/{repo}/pull-requests/{pr_id}/approve"
    bb.request("POST", path)
    typer.echo(f"Approved PR #{pr_id}")


@pr_app.command("unapprove")
def pr_unapprove(
    project: str = typer.Option(..., "--project", "-p"),
    repo: str = typer.Option(..., "--repo", "-r"),
    pr_id: int = typer.Argument(..., help="Pull request numeric ID"),
):
    """Unapprove a pull request."""
    bb = client()
    path = f"projects/{project}/repos/{repo}/pull-requests/{pr_id}/approve"
    bb.request("DELETE", path)
    typer.echo(f"Unapproved PR #{pr_id}")


@pr_app.command("decline")
def pr_decline(
    project: str = typer.Option(..., "--project", "-p"),
    repo: str = typer.Option(..., "--repo", "-r"),
    pr_id: int = typer.Argument(..., help="Pull request numeric ID"),
    version: Optional[int] = typer.Option(None, "--version", help="PR version (auto-fetched if omitted)"),
    comment: str = typer.Option("", "--comment", help="Optional decline comment"),
    json_out: bool = typer.Option(False, "--json", help="Print raw JSON response"),
):
    """Decline a pull request."""
    bb = client()
    if version is None:
        version = _get_pr_version(bb, project, repo, pr_id)
    body: Dict[str, Any] = {"version": version}
    if comment:
        body["comment"] = comment
    path = f"projects/{project}/repos/{repo}/pull-requests/{pr_id}/decline"
    resp = bb.request("POST", path, params={"version": version}, json_body=body)
    if json_out:
        _print_json(resp)
        return
    typer.echo(f"Declined PR #{pr_id}")


@pr_app.command("reopen")
def pr_reopen(
    project: str = typer.Option(..., "--project", "-p"),
    repo: str = typer.Option(..., "--repo", "-r"),
    pr_id: int = typer.Argument(..., help="Pull request numeric ID"),
    version: Optional[int] = typer.Option(None, "--version", help="PR version (auto-fetched if omitted)"),
    json_out: bool = typer.Option(False, "--json", help="Print raw JSON response"),
):
    """Re-open a pull request."""
    bb = client()
    if version is None:
        version = _get_pr_version(bb, project, repo, pr_id)
    body = {"version": version}
    path = f"projects/{project}/repos/{repo}/pull-requests/{pr_id}/reopen"
    resp = bb.request("POST", path, params={"version": version}, json_body=body)
    if json_out:
        _print_json(resp)
        return
    typer.echo(f"Re-opened PR #{pr_id}")


@pr_app.command("merge-check")
def pr_merge_check(
    project: str = typer.Option(..., "--project", "-p"),
    repo: str = typer.Option(..., "--repo", "-r"),
    pr_id: int = typer.Argument(..., help="Pull request numeric ID"),
):
    """Test if a pull request can be merged."""
    bb = client()
    path = f"projects/{project}/repos/{repo}/pull-requests/{pr_id}/merge"
    resp = bb.request("GET", path)
    _print_json(resp)


@pr_app.command("merge")
def pr_merge(
    project: str = typer.Option(..., "--project", "-p"),
    repo: str = typer.Option(..., "--repo", "-r"),
    pr_id: int = typer.Argument(..., help="Pull request numeric ID"),
    version: Optional[int] = typer.Option(None, "--version", help="PR version (auto-fetched if omitted)"),
    message: str = typer.Option("", "--message", help="Optional merge message"),
    strategy: str = typer.Option("", "--strategy", help="Merge strategy ID (if required)"),
    auto_merge: Optional[bool] = typer.Option(None, "--auto-merge/--no-auto-merge", help="Request auto-merge"),
    auto_subject: str = typer.Option("", "--auto-subject", help="Optional auto-merge subject"),
    json_out: bool = typer.Option(False, "--json", help="Print raw JSON response"),
):
    """Merge a pull request."""
    bb = client()
    if version is None:
        version = _get_pr_version(bb, project, repo, pr_id)
    body: Dict[str, Any] = {"version": version}
    if message:
        body["message"] = message
    if strategy:
        body["strategyId"] = strategy
    if auto_merge is not None:
        body["autoMerge"] = bool(auto_merge)
    if auto_subject:
        body["autoSubject"] = auto_subject
    path = f"projects/{project}/repos/{repo}/pull-requests/{pr_id}/merge"
    resp = bb.request("POST", path, params={"version": version}, json_body=body)
    if json_out:
        _print_json(resp)
        return
    typer.echo(f"Merged PR #{pr_id}")


@pr_app.command("update")
def pr_update(
    project: str = typer.Option(..., "--project", "-p"),
    repo: str = typer.Option(..., "--repo", "-r"),
    pr_id: int = typer.Argument(..., help="Pull request numeric ID"),
    version: Optional[int] = typer.Option(None, "--version", help="PR version (auto-fetched if omitted)"),
    title: Optional[str] = typer.Option(None, "--title", help="New title"),
    description: Optional[str] = typer.Option(None, "--description", help="New description (use empty string to clear)"),
    reviewer: List[str] = typer.Option(
        [],
        "--reviewer",
        help="Reviewer username (repeatable). If set, replaces reviewers list.",
    ),
    draft: Optional[bool] = typer.Option(None, "--draft/--no-draft", help="Set PR draft status"),
    json_out: bool = typer.Option(False, "--json", help="Print raw JSON response"),
):
    """Update pull request metadata (title/description/reviewers/draft)."""
    if title is None and description is None and not reviewer and draft is None:
        raise BBError("Nothing to update. Provide --title, --description, --reviewer, or --draft/--no-draft.")
    bb = client()
    if version is None:
        version = _get_pr_version(bb, project, repo, pr_id)
    body: Dict[str, Any] = {"version": version}
    if title is not None:
        body["title"] = title
    if description is not None:
        body["description"] = description
    if reviewer:
        body["reviewers"] = [{"user": {"name": r}} for r in reviewer]
    if draft is not None:
        body["draft"] = bool(draft)
    path = f"projects/{project}/repos/{repo}/pull-requests/{pr_id}"
    resp = bb.request("PUT", path, json_body=body)
    if json_out:
        _print_json(resp)
        return
    new_version = resp.get("version", "?")
    typer.echo(f"Updated PR #{pr_id} (version {new_version})")


@pr_app.command("watch")
def pr_watch(
    project: str = typer.Option(..., "--project", "-p"),
    repo: str = typer.Option(..., "--repo", "-r"),
    pr_id: int = typer.Argument(..., help="Pull request numeric ID"),
):
    """Watch a pull request."""
    bb = client()
    path = f"projects/{project}/repos/{repo}/pull-requests/{pr_id}/watch"
    bb.request("POST", path)
    typer.echo(f"Watching PR #{pr_id}")


@pr_app.command("unwatch")
def pr_unwatch(
    project: str = typer.Option(..., "--project", "-p"),
    repo: str = typer.Option(..., "--repo", "-r"),
    pr_id: int = typer.Argument(..., help="Pull request numeric ID"),
):
    """Stop watching a pull request."""
    bb = client()
    path = f"projects/{project}/repos/{repo}/pull-requests/{pr_id}/watch"
    bb.request("DELETE", path)
    typer.echo(f"Stopped watching PR #{pr_id}")


@participants_app.command("list")
def pr_participants_list(
    project: str = typer.Option(..., "--project", "-p"),
    repo: str = typer.Option(..., "--repo", "-r"),
    pr_id: int = typer.Argument(..., help="Pull request numeric ID"),
    limit: int = typer.Option(50, help="Page size"),
    max_items: int = typer.Option(200, help="Max items to fetch across pages"),
    json_out: bool = typer.Option(False, "--json", help="Print raw JSON instead of a table"),
):
    """List participants/reviewers on a pull request."""
    bb = client()
    path = f"projects/{project}/repos/{repo}/pull-requests/{pr_id}/participants"
    participants = bb.paged_get(path, params={}, limit=limit, max_items=max_items)
    if json_out:
        _print_json(participants)
    else:
        _print_participants(participants)


@participants_app.command("add")
def pr_participants_add(
    project: str = typer.Option(..., "--project", "-p"),
    repo: str = typer.Option(..., "--repo", "-r"),
    pr_id: int = typer.Argument(..., help="Pull request numeric ID"),
    user: str = typer.Option(..., "--user", "-u", help="Username or user slug"),
    role: str = typer.Option("REVIEWER", "--role", help="AUTHOR, REVIEWER, or PARTICIPANT"),
    json_out: bool = typer.Option(False, "--json", help="Print raw JSON response"),
):
    """Assign a participant role (use role REVIEWER to add a reviewer)."""
    bb = client()
    role = _norm_choice(role, ROLE_CHOICES, "role")
    body = {"user": {"name": user}, "role": role}
    path = f"projects/{project}/repos/{repo}/pull-requests/{pr_id}/participants"
    resp = bb.request("POST", path, json_body=body)
    if json_out:
        _print_json(resp)
        return
    typer.echo(f"Added {user} as {role} on PR #{pr_id}")


@participants_app.command("remove")
def pr_participants_remove(
    project: str = typer.Option(..., "--project", "-p"),
    repo: str = typer.Option(..., "--repo", "-r"),
    pr_id: int = typer.Argument(..., help="Pull request numeric ID"),
    user: str = typer.Argument(..., help="User slug to remove"),
):
    """Remove a participant/reviewer from a pull request."""
    bb = client()
    path = f"projects/{project}/repos/{repo}/pull-requests/{pr_id}/participants/{user}"
    bb.request("DELETE", path)
    typer.echo(f"Removed {user} from PR #{pr_id}")


@participants_app.command("status")
def pr_participants_status(
    project: str = typer.Option(..., "--project", "-p"),
    repo: str = typer.Option(..., "--repo", "-r"),
    pr_id: int = typer.Argument(..., help="Pull request numeric ID"),
    user: str = typer.Argument(..., help="User slug to update"),
    status: str = typer.Option(..., "--status", help="UNAPPROVED, NEEDS_WORK, or APPROVED"),
    last_reviewed_commit: Optional[str] = typer.Option(
        None, "--last-reviewed-commit", help="Optional commit hash last reviewed"
    ),
    version: Optional[int] = typer.Option(None, "--version", help="PR version (optional)"),
    json_out: bool = typer.Option(False, "--json", help="Print raw JSON response"),
):
    """Change a participant's status on a pull request."""
    bb = client()
    status = _norm_choice(status, STATUS_CHOICES, "status")
    body: Dict[str, Any] = {"status": status}
    if last_reviewed_commit:
        body["lastReviewedCommit"] = last_reviewed_commit
    params = {"version": version} if version is not None else None
    path = f"projects/{project}/repos/{repo}/pull-requests/{pr_id}/participants/{user}"
    resp = bb.request("PUT", path, params=params, json_body=body)
    if json_out:
        _print_json(resp)
        return
    typer.echo(f"Updated {user} status to {status} on PR #{pr_id}")


@participants_app.command("search")
def pr_participants_search(
    project: str = typer.Option(..., "--project", "-p"),
    repo: str = typer.Option(..., "--repo", "-r"),
    filter: Optional[str] = typer.Option(None, "--filter", help="Filter by username, name, or email"),
    role: Optional[str] = typer.Option(None, "--role", help="AUTHOR, REVIEWER, or PARTICIPANT"),
    direction: Optional[str] = typer.Option(None, "--direction", help="INCOMING or OUTGOING"),
    limit: int = typer.Option(50, help="Page size"),
    max_items: int = typer.Option(200, help="Max items to fetch across pages"),
    json_out: bool = typer.Option(False, "--json", help="Print raw JSON response"),
):
    """Search pull request participants in a repository."""
    bb = client()
    params: Dict[str, Any] = {}
    if filter:
        params["filter"] = filter
    if role:
        params["role"] = _norm_choice(role, ROLE_CHOICES, "role")
    if direction:
        params["direction"] = direction
    path = f"projects/{project}/repos/{repo}/participants"
    participants = bb.paged_get(path, params=params, limit=limit, max_items=max_items)
    if json_out:
        _print_json(participants)
    else:
        _print_participants(participants)


@comments_app.command("add")
def pr_comments_add(
    project: str = typer.Option(..., "--project", "-p"),
    repo: str = typer.Option(..., "--repo", "-r"),
    pr_id: int = typer.Argument(..., help="Pull request numeric ID"),
    text: str = typer.Option(..., "--text", "-t", help="Comment text"),
):
    """Add a comment to a pull request."""
    pr_comment(project=project, repo=repo, pr_id=pr_id, text=text)


@comments_app.command("list")
def pr_comments_list(
    project: str = typer.Option(..., "--project", "-p"),
    repo: str = typer.Option(..., "--repo", "-r"),
    pr_id: int = typer.Argument(..., help="Pull request numeric ID"),
    file_path: str = typer.Option(..., "--path", help="File path to stream comments for"),
    from_hash: Optional[str] = typer.Option(None, "--from-hash", help="From commit hash"),
    to_hash: Optional[str] = typer.Option(None, "--to-hash", help="To commit hash"),
    diff_types: Optional[str] = typer.Option(None, "--diff-types", help="Comma-separated diff types"),
    states: Optional[str] = typer.Option(None, "--states", help="Comma-separated states (OPEN,RESOLVED)"),
    anchor_state: Optional[str] = typer.Option(None, "--anchor-state", help="ACTIVE, ORPHANED, or ALL"),
    limit: int = typer.Option(50, help="Page size"),
    max_items: int = typer.Option(200, help="Max items to fetch across pages"),
    json_out: bool = typer.Option(False, "--json", help="Print raw JSON response"),
):
    """List pull request comments for a file path."""
    bb = client()
    params: Dict[str, Any] = {"path": file_path}
    if from_hash:
        params["fromHash"] = from_hash
    if to_hash:
        params["toHash"] = to_hash
    if diff_types:
        params["diffTypes"] = diff_types
    if states:
        params["states"] = states
    if anchor_state:
        params["anchorState"] = anchor_state
    path = f"projects/{project}/repos/{repo}/pull-requests/{pr_id}/comments"
    comments = bb.paged_get(path, params=params, limit=limit, max_items=max_items)
    if json_out:
        _print_json(comments)
    else:
        _print_json(comments)


@comments_app.command("get")
def pr_comments_get(
    project: str = typer.Option(..., "--project", "-p"),
    repo: str = typer.Option(..., "--repo", "-r"),
    pr_id: int = typer.Argument(..., help="Pull request numeric ID"),
    comment_id: int = typer.Argument(..., help="Comment ID"),
):
    """Get a pull request comment."""
    bb = client()
    path = f"projects/{project}/repos/{repo}/pull-requests/{pr_id}/comments/{comment_id}"
    resp = bb.request("GET", path)
    _print_json(resp)


@comments_app.command("update")
def pr_comments_update(
    project: str = typer.Option(..., "--project", "-p"),
    repo: str = typer.Option(..., "--repo", "-r"),
    pr_id: int = typer.Argument(..., help="Pull request numeric ID"),
    comment_id: int = typer.Argument(..., help="Comment ID"),
    text: Optional[str] = typer.Option(None, "--text", help="Updated comment text"),
    severity: Optional[str] = typer.Option(None, "--severity", help="NORMAL or BLOCKER"),
    state: Optional[str] = typer.Option(None, "--state", help="OPEN or RESOLVED"),
    version: Optional[int] = typer.Option(None, "--version", help="Comment version (auto-fetched if omitted)"),
    json_out: bool = typer.Option(False, "--json", help="Print raw JSON response"),
):
    """Update a pull request comment."""
    if text is None and severity is None and state is None:
        raise BBError("Nothing to update. Provide --text, --severity, or --state.")
    bb = client()
    if version is None:
        version = _get_comment_version(bb, project, repo, pr_id, comment_id)
    body: Dict[str, Any] = {"version": version}
    if text is not None:
        body["text"] = text
    if severity is not None:
        body["severity"] = _norm_choice(severity, COMMENT_SEVERITY_CHOICES, "severity")
    if state is not None:
        body["state"] = _norm_choice(state, COMMENT_STATE_CHOICES, "state")
    path = f"projects/{project}/repos/{repo}/pull-requests/{pr_id}/comments/{comment_id}"
    resp = bb.request("PUT", path, json_body=body)
    if json_out:
        _print_json(resp)
    else:
        typer.echo(f"Updated comment {comment_id} on PR #{pr_id}")


@comments_app.command("delete")
def pr_comments_delete(
    project: str = typer.Option(..., "--project", "-p"),
    repo: str = typer.Option(..., "--repo", "-r"),
    pr_id: int = typer.Argument(..., help="Pull request numeric ID"),
    comment_id: int = typer.Argument(..., help="Comment ID"),
    version: Optional[int] = typer.Option(None, "--version", help="Comment version (auto-fetched if omitted)"),
):
    """Delete a pull request comment."""
    bb = client()
    if version is None:
        version = _get_comment_version(bb, project, repo, pr_id, comment_id)
    path = f"projects/{project}/repos/{repo}/pull-requests/{pr_id}/comments/{comment_id}"
    bb.request("DELETE", path, params={"version": version})
    typer.echo(f"Deleted comment {comment_id} on PR #{pr_id}")


@comments_app.command("apply-suggestion")
def pr_comments_apply_suggestion(
    project: str = typer.Option(..., "--project", "-p"),
    repo: str = typer.Option(..., "--repo", "-r"),
    pr_id: int = typer.Argument(..., help="Pull request numeric ID"),
    comment_id: int = typer.Argument(..., help="Comment ID"),
    suggestion_index: int = typer.Option(..., "--suggestion-index", help="Suggestion index"),
    comment_version: Optional[int] = typer.Option(None, "--comment-version", help="Comment version"),
    pr_version: Optional[int] = typer.Option(None, "--pr-version", help="Pull request version"),
    commit_message: str = typer.Option("", "--commit-message", help="Optional commit message"),
    json_out: bool = typer.Option(False, "--json", help="Print raw JSON response"),
):
    """Apply a suggestion from a pull request comment."""
    bb = client()
    if comment_version is None:
        comment_version = _get_comment_version(bb, project, repo, pr_id, comment_id)
    if pr_version is None:
        pr_version = _get_pr_version(bb, project, repo, pr_id)
    body: Dict[str, Any] = {
        "commentVersion": comment_version,
        "pullRequestVersion": pr_version,
        "suggestionIndex": suggestion_index,
    }
    if commit_message:
        body["commitMessage"] = commit_message
    path = (
        f"projects/{project}/repos/{repo}/pull-requests/{pr_id}/comments/{comment_id}/apply-suggestion"
    )
    resp = bb.request("POST", path, json_body=body)
    if json_out:
        _print_json(resp)
    else:
        typer.echo(f"Applied suggestion from comment {comment_id} on PR #{pr_id}")


@comments_app.command("react")
def pr_comments_react(
    project: str = typer.Option(..., "--project", "-p"),
    repo: str = typer.Option(..., "--repo", "-r"),
    pr_id: int = typer.Argument(..., help="Pull request numeric ID"),
    comment_id: int = typer.Argument(..., help="Comment ID"),
    emoticon: str = typer.Option(..., "--emoticon", help="Reaction emoticon (e.g. :+1:)"),
):
    """React to a pull request comment."""
    bb = client()
    path = (
        f"comment-likes/latest/projects/{project}/repos/{repo}/pull-requests/{pr_id}"
        f"/comments/{comment_id}/reactions/{emoticon}"
    )
    bb.request_rest("PUT", path)
    typer.echo(f"Reacted to comment {comment_id} on PR #{pr_id}")


@comments_app.command("unreact")
def pr_comments_unreact(
    project: str = typer.Option(..., "--project", "-p"),
    repo: str = typer.Option(..., "--repo", "-r"),
    pr_id: int = typer.Argument(..., help="Pull request numeric ID"),
    comment_id: int = typer.Argument(..., help="Comment ID"),
    emoticon: str = typer.Option(..., "--emoticon", help="Reaction emoticon (e.g. :+1:)"),
):
    """Remove a reaction from a pull request comment."""
    bb = client()
    path = (
        f"comment-likes/latest/projects/{project}/repos/{repo}/pull-requests/{pr_id}"
        f"/comments/{comment_id}/reactions/{emoticon}"
    )
    bb.request_rest("DELETE", path)
    typer.echo(f"Removed reaction from comment {comment_id} on PR #{pr_id}")


@blockers_app.command("list")
def pr_blockers_list(
    project: str = typer.Option(..., "--project", "-p"),
    repo: str = typer.Option(..., "--repo", "-r"),
    pr_id: int = typer.Argument(..., help="Pull request numeric ID"),
    states: Optional[str] = typer.Option(None, "--states", help="Comma-separated states (OPEN,RESOLVED)"),
    count: bool = typer.Option(False, "--count", help="Return counts only"),
    limit: int = typer.Option(50, help="Page size"),
    max_items: int = typer.Option(200, help="Max items to fetch across pages"),
    json_out: bool = typer.Option(False, "--json", help="Print raw JSON response"),
):
    """List blocker comments for a pull request."""
    bb = client()
    params: Dict[str, Any] = {}
    if states:
        params["states"] = states
    if count:
        params["count"] = "true"
        resp = bb.request(
            "GET", f"projects/{project}/repos/{repo}/pull-requests/{pr_id}/blocker-comments", params=params
        )
        if json_out:
            _print_json(resp)
        else:
            _print_json(resp)
        return
    path = f"projects/{project}/repos/{repo}/pull-requests/{pr_id}/blocker-comments"
    comments = bb.paged_get(path, params=params, limit=limit, max_items=max_items)
    if json_out:
        _print_json(comments)
    else:
        _print_json(comments)


@blockers_app.command("add")
def pr_blockers_add(
    project: str = typer.Option(..., "--project", "-p"),
    repo: str = typer.Option(..., "--repo", "-r"),
    pr_id: int = typer.Argument(..., help="Pull request numeric ID"),
    text: str = typer.Option(..., "--text", "-t", help="Comment text"),
    json_out: bool = typer.Option(False, "--json", help="Print raw JSON response"),
):
    """Add a blocker comment to a pull request."""
    bb = client()
    path = f"projects/{project}/repos/{repo}/pull-requests/{pr_id}/blocker-comments"
    body = {"text": text}
    resp = bb.request("POST", path, json_body=body)
    if json_out:
        _print_json(resp)
    else:
        comment_id = resp.get("id", "?")
        typer.echo(f"Added blocker comment {comment_id} to PR #{pr_id}")


@blockers_app.command("get")
def pr_blockers_get(
    project: str = typer.Option(..., "--project", "-p"),
    repo: str = typer.Option(..., "--repo", "-r"),
    pr_id: int = typer.Argument(..., help="Pull request numeric ID"),
    comment_id: int = typer.Argument(..., help="Comment ID"),
):
    """Get a blocker comment."""
    bb = client()
    path = f"projects/{project}/repos/{repo}/pull-requests/{pr_id}/blocker-comments/{comment_id}"
    resp = bb.request("GET", path)
    _print_json(resp)


@blockers_app.command("update")
def pr_blockers_update(
    project: str = typer.Option(..., "--project", "-p"),
    repo: str = typer.Option(..., "--repo", "-r"),
    pr_id: int = typer.Argument(..., help="Pull request numeric ID"),
    comment_id: int = typer.Argument(..., help="Comment ID"),
    text: Optional[str] = typer.Option(None, "--text", help="Updated comment text"),
    severity: Optional[str] = typer.Option(None, "--severity", help="NORMAL or BLOCKER"),
    state: Optional[str] = typer.Option(None, "--state", help="OPEN or RESOLVED"),
    version: Optional[int] = typer.Option(None, "--version", help="Comment version (auto-fetched if omitted)"),
    json_out: bool = typer.Option(False, "--json", help="Print raw JSON response"),
):
    """Update a blocker comment."""
    if text is None and severity is None and state is None:
        raise BBError("Nothing to update. Provide --text, --severity, or --state.")
    bb = client()
    if version is None:
        version = _get_comment_version(bb, project, repo, pr_id, comment_id, blocker=True)
    body: Dict[str, Any] = {"version": version}
    if text is not None:
        body["text"] = text
    if severity is not None:
        body["severity"] = _norm_choice(severity, COMMENT_SEVERITY_CHOICES, "severity")
    if state is not None:
        body["state"] = _norm_choice(state, COMMENT_STATE_CHOICES, "state")
    path = f"projects/{project}/repos/{repo}/pull-requests/{pr_id}/blocker-comments/{comment_id}"
    resp = bb.request("PUT", path, json_body=body)
    if json_out:
        _print_json(resp)
    else:
        typer.echo(f"Updated blocker comment {comment_id} on PR #{pr_id}")


@blockers_app.command("delete")
def pr_blockers_delete(
    project: str = typer.Option(..., "--project", "-p"),
    repo: str = typer.Option(..., "--repo", "-r"),
    pr_id: int = typer.Argument(..., help="Pull request numeric ID"),
    comment_id: int = typer.Argument(..., help="Comment ID"),
    version: Optional[int] = typer.Option(None, "--version", help="Comment version (auto-fetched if omitted)"),
):
    """Delete a blocker comment."""
    bb = client()
    if version is None:
        version = _get_comment_version(bb, project, repo, pr_id, comment_id, blocker=True)
    path = f"projects/{project}/repos/{repo}/pull-requests/{pr_id}/blocker-comments/{comment_id}"
    bb.request("DELETE", path, params={"version": version})
    typer.echo(f"Deleted blocker comment {comment_id} on PR #{pr_id}")


@review_app.command("get")
def pr_review_get(
    project: str = typer.Option(..., "--project", "-p"),
    repo: str = typer.Option(..., "--repo", "-r"),
    pr_id: int = typer.Argument(..., help="Pull request numeric ID"),
):
    """Get pull request review thread."""
    bb = client()
    path = f"projects/{project}/repos/{repo}/pull-requests/{pr_id}/review"
    resp = bb.request("GET", path)
    _print_json(resp)


@review_app.command("complete")
def pr_review_complete(
    project: str = typer.Option(..., "--project", "-p"),
    repo: str = typer.Option(..., "--repo", "-r"),
    pr_id: int = typer.Argument(..., help="Pull request numeric ID"),
    comment_text: Optional[str] = typer.Option(None, "--comment", help="General review comment"),
    last_reviewed_commit: Optional[str] = typer.Option(
        None, "--last-reviewed-commit", help="Commit hash last reviewed"
    ),
    status: Optional[str] = typer.Option(None, "--status", help="UNAPPROVED, NEEDS_WORK, or APPROVED"),
    json_out: bool = typer.Option(False, "--json", help="Print raw JSON response"),
):
    """Complete a pull request review."""
    if comment_text is None and last_reviewed_commit is None and status is None:
        raise BBError("Nothing to update. Provide --comment, --last-reviewed-commit, or --status.")
    bb = client()
    body: Dict[str, Any] = {}
    if comment_text is not None:
        body["commentText"] = comment_text
    if last_reviewed_commit is not None:
        body["lastReviewedCommit"] = last_reviewed_commit
    if status is not None:
        body["participantStatus"] = _norm_choice(status, STATUS_CHOICES, "status")
    path = f"projects/{project}/repos/{repo}/pull-requests/{pr_id}/review"
    resp = bb.request("PUT", path, json_body=body)
    if json_out:
        _print_json(resp)
    else:
        typer.echo(f"Completed review for PR #{pr_id}")


@review_app.command("discard")
def pr_review_discard(
    project: str = typer.Option(..., "--project", "-p"),
    repo: str = typer.Option(..., "--repo", "-r"),
    pr_id: int = typer.Argument(..., help="Pull request numeric ID"),
):
    """Discard pull request review."""
    bb = client()
    path = f"projects/{project}/repos/{repo}/pull-requests/{pr_id}/review"
    bb.request("DELETE", path)
    typer.echo(f"Discarded review for PR #{pr_id}")


@auto_merge_app.command("get")
def pr_auto_merge_get(
    project: str = typer.Option(..., "--project", "-p"),
    repo: str = typer.Option(..., "--repo", "-r"),
    pr_id: int = typer.Argument(..., help="Pull request numeric ID"),
):
    """Get auto-merge request for a pull request."""
    bb = client()
    path = f"projects/{project}/repos/{repo}/pull-requests/{pr_id}/auto-merge"
    resp = bb.request("GET", path)
    _print_json(resp)


@auto_merge_app.command("set")
def pr_auto_merge_set(
    project: str = typer.Option(..., "--project", "-p"),
    repo: str = typer.Option(..., "--repo", "-r"),
    pr_id: int = typer.Argument(..., help="Pull request numeric ID"),
):
    """Request auto-merge for a pull request."""
    bb = client()
    path = f"projects/{project}/repos/{repo}/pull-requests/{pr_id}/auto-merge"
    bb.request("POST", path)
    typer.echo(f"Requested auto-merge for PR #{pr_id}")


@auto_merge_app.command("cancel")
def pr_auto_merge_cancel(
    project: str = typer.Option(..., "--project", "-p"),
    repo: str = typer.Option(..., "--repo", "-r"),
    pr_id: int = typer.Argument(..., help="Pull request numeric ID"),
):
    """Cancel auto-merge for a pull request."""
    bb = client()
    path = f"projects/{project}/repos/{repo}/pull-requests/{pr_id}/auto-merge"
    bb.request("DELETE", path)
    typer.echo(f"Cancelled auto-merge for PR #{pr_id}")


@pr_app.command("activities")
def pr_activities(
    project: str = typer.Option(..., "--project", "-p"),
    repo: str = typer.Option(..., "--repo", "-r"),
    pr_id: int = typer.Argument(..., help="Pull request numeric ID"),
    from_id: Optional[str] = typer.Option(None, "--from-id", help="Start from activity/comment id"),
    from_type: Optional[str] = typer.Option(None, "--from-type", help="COMMENT or ACTIVITY"),
    limit: int = typer.Option(50, help="Page size"),
    max_items: int = typer.Option(200, help="Max items to fetch across pages"),
    json_out: bool = typer.Option(False, "--json", help="Print raw JSON response"),
):
    """Get pull request activity."""
    bb = client()
    params: Dict[str, Any] = {}
    if from_id:
        params["fromId"] = from_id
    if from_type:
        params["fromType"] = from_type
    path = f"projects/{project}/repos/{repo}/pull-requests/{pr_id}/activities"
    activities = bb.paged_get(path, params=params, limit=limit, max_items=max_items)
    if json_out:
        _print_json(activities)
    else:
        _print_json(activities)


@pr_app.command("changes")
def pr_changes(
    project: str = typer.Option(..., "--project", "-p"),
    repo: str = typer.Option(..., "--repo", "-r"),
    pr_id: int = typer.Argument(..., help="Pull request numeric ID"),
    change_scope: Optional[str] = typer.Option(None, "--change-scope", help="ALL, UNREVIEWED, or RANGE"),
    since_id: Optional[str] = typer.Option(None, "--since-id", help="Since commit hash (for RANGE)"),
    until_id: Optional[str] = typer.Option(None, "--until-id", help="Until commit hash (for RANGE)"),
    with_comments: Optional[bool] = typer.Option(
        None, "--with-comments/--no-with-comments", help="Include comment counts"
    ),
    limit: int = typer.Option(50, help="Page size"),
    max_items: int = typer.Option(200, help="Max items to fetch across pages"),
    json_out: bool = typer.Option(False, "--json", help="Print raw JSON response"),
):
    """Get pull request changes."""
    bb = client()
    params: Dict[str, Any] = {}
    if change_scope:
        params["changeScope"] = change_scope
    if since_id:
        params["sinceId"] = since_id
    if until_id:
        params["untilId"] = until_id
    if with_comments is not None:
        params["withComments"] = str(with_comments).lower()
    path = f"projects/{project}/repos/{repo}/pull-requests/{pr_id}/changes"
    changes = bb.paged_get(path, params=params, limit=limit, max_items=max_items)
    if json_out:
        _print_json(changes)
    else:
        _print_json(changes)


@pr_app.command("commits")
def pr_commits(
    project: str = typer.Option(..., "--project", "-p"),
    repo: str = typer.Option(..., "--repo", "-r"),
    pr_id: int = typer.Argument(..., help="Pull request numeric ID"),
    with_counts: Optional[bool] = typer.Option(
        None, "--with-counts/--no-with-counts", help="Include author/total counts"
    ),
    avatar_size: Optional[int] = typer.Option(None, "--avatar-size", help="Avatar size in pixels"),
    avatar_scheme: Optional[str] = typer.Option(None, "--avatar-scheme", help="Avatar scheme (http/https)"),
    limit: int = typer.Option(50, help="Page size"),
    max_items: int = typer.Option(200, help="Max items to fetch across pages"),
    json_out: bool = typer.Option(False, "--json", help="Print raw JSON response"),
):
    """Get pull request commits."""
    bb = client()
    params: Dict[str, Any] = {}
    if with_counts is not None:
        params["withCounts"] = str(with_counts).lower()
    if avatar_size is not None:
        params["avatarSize"] = avatar_size
    if avatar_scheme:
        params["avatarScheme"] = avatar_scheme
    path = f"projects/{project}/repos/{repo}/pull-requests/{pr_id}/commits"
    commits = bb.paged_get(path, params=params, limit=limit, max_items=max_items)
    if json_out:
        _print_json(commits)
    else:
        _print_json(commits)


@pr_app.command("diff")
def pr_diff(
    project: str = typer.Option(..., "--project", "-p"),
    repo: str = typer.Option(..., "--repo", "-r"),
    pr_id: int = typer.Argument(..., help="Pull request numeric ID"),
    context_lines: Optional[int] = typer.Option(None, "--context-lines", help="Context lines"),
    whitespace: Optional[str] = typer.Option(None, "--whitespace", help="Whitespace option (ignore-all)"),
):
    """Stream raw pull request diff."""
    bb = client()
    params: Dict[str, Any] = {}
    if context_lines is not None:
        params["contextLines"] = context_lines
    if whitespace:
        params["whitespace"] = whitespace
    path = f"projects/{project}/repos/{repo}/pull-requests/{pr_id}.diff"
    resp = bb.request("GET", path, params=params)
    _print_raw(resp)


@pr_app.command("diff-file")
def pr_diff_file(
    project: str = typer.Option(..., "--project", "-p"),
    repo: str = typer.Option(..., "--repo", "-r"),
    pr_id: int = typer.Argument(..., help="Pull request numeric ID"),
    file_path: str = typer.Argument(..., help="File path within repository"),
    since_id: Optional[str] = typer.Option(None, "--since-id", help="Since commit hash"),
    until_id: Optional[str] = typer.Option(None, "--until-id", help="Until commit hash"),
    src_path: Optional[str] = typer.Option(None, "--src-path", help="Previous path (rename/copy)"),
    diff_type: Optional[str] = typer.Option(None, "--diff-type", help="Diff type hint"),
    context_lines: Optional[int] = typer.Option(None, "--context-lines", help="Context lines"),
    whitespace: Optional[str] = typer.Option(None, "--whitespace", help="Whitespace option (ignore-all)"),
    with_comments: Optional[bool] = typer.Option(
        None, "--with-comments/--no-with-comments", help="Include comments in diff"
    ),
    avatar_size: Optional[int] = typer.Option(None, "--avatar-size", help="Avatar size in pixels"),
    avatar_scheme: Optional[str] = typer.Option(None, "--avatar-scheme", help="Avatar scheme (http/https)"),
):
    """Stream a diff for a file within a pull request."""
    bb = client()
    params: Dict[str, Any] = {}
    if since_id:
        params["sinceId"] = since_id
    if until_id:
        params["untilId"] = until_id
    if src_path:
        params["srcPath"] = src_path
    if diff_type:
        params["diffType"] = diff_type
    if context_lines is not None:
        params["contextLines"] = context_lines
    if whitespace:
        params["whitespace"] = whitespace
    if with_comments is not None:
        params["withComments"] = str(with_comments).lower()
    if avatar_size is not None:
        params["avatarSize"] = avatar_size
    if avatar_scheme:
        params["avatarScheme"] = avatar_scheme
    enc_path = _encode_path(file_path)
    path = f"projects/{project}/repos/{repo}/pull-requests/{pr_id}/diff/{enc_path}"
    resp = bb.request("GET", path, params=params)
    _print_raw(resp)


@pr_app.command("diff-stats")
def pr_diff_stats(
    project: str = typer.Option(..., "--project", "-p"),
    repo: str = typer.Option(..., "--repo", "-r"),
    pr_id: int = typer.Argument(..., help="Pull request numeric ID"),
    file_path: str = typer.Argument(..., help="File path within repository"),
    since_id: Optional[str] = typer.Option(None, "--since-id", help="Since commit hash"),
    until_id: Optional[str] = typer.Option(None, "--until-id", help="Until commit hash"),
    src_path: Optional[str] = typer.Option(None, "--src-path", help="Previous path (rename/copy)"),
    whitespace: Optional[str] = typer.Option(None, "--whitespace", help="Whitespace option (ignore-all)"),
):
    """Get diff stats summary for a file within a pull request."""
    bb = client()
    params: Dict[str, Any] = {}
    if since_id:
        params["sinceId"] = since_id
    if until_id:
        params["untilId"] = until_id
    if src_path:
        params["srcPath"] = src_path
    if whitespace:
        params["whitespace"] = whitespace
    enc_path = _encode_path(file_path)
    path = f"projects/{project}/repos/{repo}/pull-requests/{pr_id}/diff-stats-summary/{enc_path}"
    resp = bb.request("GET", path, params=params)
    _print_json(resp)


@pr_app.command("patch")
def pr_patch(
    project: str = typer.Option(..., "--project", "-p"),
    repo: str = typer.Option(..., "--repo", "-r"),
    pr_id: int = typer.Argument(..., help="Pull request numeric ID"),
):
    """Stream pull request as a patch."""
    bb = client()
    path = f"projects/{project}/repos/{repo}/pull-requests/{pr_id}.patch"
    resp = bb.request("GET", path)
    _print_raw(resp)


@pr_app.command("merge-base")
def pr_merge_base(
    project: str = typer.Option(..., "--project", "-p"),
    repo: str = typer.Option(..., "--repo", "-r"),
    pr_id: int = typer.Argument(..., help="Pull request numeric ID"),
):
    """Get the merge base for a pull request."""
    bb = client()
    path = f"projects/{project}/repos/{repo}/pull-requests/{pr_id}/merge-base"
    resp = bb.request("GET", path)
    _print_json(resp)


@pr_app.command("commit-message")
def pr_commit_message(
    project: str = typer.Option(..., "--project", "-p"),
    repo: str = typer.Option(..., "--repo", "-r"),
    pr_id: int = typer.Argument(..., help="Pull request numeric ID"),
):
    """Get commit message suggestion for a pull request."""
    bb = client()
    path = f"projects/{project}/repos/{repo}/pull-requests/{pr_id}/commit-message-suggestion"
    resp = bb.request("GET", path)
    _print_json(resp)


@pr_app.command("rebase-check")
def pr_rebase_check(
    project: str = typer.Option(..., "--project", "-p"),
    repo: str = typer.Option(..., "--repo", "-r"),
    pr_id: int = typer.Argument(..., help="Pull request numeric ID"),
):
    """Check whether a pull request can be rebased."""
    bb = client()
    path = f"git/latest/projects/{project}/repos/{repo}/pull-requests/{pr_id}/rebase"
    resp = bb.request_rest("GET", path)
    _print_json(resp)


@pr_app.command("rebase")
def pr_rebase(
    project: str = typer.Option(..., "--project", "-p"),
    repo: str = typer.Option(..., "--repo", "-r"),
    pr_id: int = typer.Argument(..., help="Pull request numeric ID"),
    version: Optional[int] = typer.Option(None, "--version", help="PR version (auto-fetched if omitted)"),
    json_out: bool = typer.Option(False, "--json", help="Print raw JSON response"),
):
    """Rebase a pull request."""
    bb = client()
    if version is None:
        version = _get_pr_version(bb, project, repo, pr_id)
    body = {"version": version}
    path = f"git/latest/projects/{project}/repos/{repo}/pull-requests/{pr_id}/rebase"
    resp = bb.request_rest("POST", path, json_body=body)
    if json_out:
        _print_json(resp)
    else:
        typer.echo(f"Rebased PR #{pr_id}")


@pr_app.command("delete")
def pr_delete(
    project: str = typer.Option(..., "--project", "-p"),
    repo: str = typer.Option(..., "--repo", "-r"),
    pr_id: int = typer.Argument(..., help="Pull request numeric ID"),
    version: Optional[int] = typer.Option(None, "--version", help="PR version (auto-fetched if omitted)"),
    json_out: bool = typer.Option(False, "--json", help="Print raw JSON response"),
):
    """Delete a pull request."""
    bb = client()
    if version is None:
        version = _get_pr_version(bb, project, repo, pr_id)
    body = {"version": version}
    path = f"projects/{project}/repos/{repo}/pull-requests/{pr_id}"
    resp = bb.request("DELETE", path, json_body=body)
    if json_out:
        _print_json(resp)
    else:
        typer.echo(f"Deleted PR #{pr_id}")


@pr_app.command("for-commit")
def pr_for_commit(
    project: str = typer.Option(..., "--project", "-p"),
    repo: str = typer.Option(..., "--repo", "-r"),
    commit_id: str = typer.Argument(..., help="Commit ID"),
    limit: int = typer.Option(50, help="Page size"),
    max_items: int = typer.Option(200, help="Max items to fetch across pages"),
    json_out: bool = typer.Option(False, "--json", help="Print raw JSON response"),
):
    """Get pull requests containing a commit."""
    bb = client()
    path = f"projects/{project}/repos/{repo}/commits/{commit_id}/pull-requests"
    prs = bb.paged_get(path, params={}, limit=limit, max_items=max_items)
    if json_out:
        _print_json(prs)
    else:
        _print_json(prs)


@app.command("doctor")
def doctor():
    """Sanity checks: validates env vars and hits a lightweight endpoint."""
    bb = client()
    # Hit an endpoint that typically requires only auth and returns quickly.
    # We'll use dashboard pull-requests (even if empty) as a general check.
    resp = bb.request("GET", "dashboard/pull-requests", params={"limit": 1, "start": 0})
    # If we got here, auth + base URL are OK.
    typer.echo("OK: BITBUCKET_SERVER and BITBUCKET_API_TOKEN look usable.")
    if isinstance(resp, dict) and "values" in resp:
        typer.echo(f"Dashboard PRs visible: {len(resp.get('values') or [])} item(s) on first page.")


def main() -> None:
    try:
        app()
    except BBError as e:
        typer.secho(str(e), fg=typer.colors.RED, err=True)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
