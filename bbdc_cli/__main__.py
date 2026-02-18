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
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional
from urllib.parse import quote

import requests
import typer

app = typer.Typer(no_args_is_help=True, add_completion=False)
account_app = typer.Typer(no_args_is_help=True, add_completion=False)
pr_app = typer.Typer(no_args_is_help=True, add_completion=False)
participants_app = typer.Typer(no_args_is_help=True, add_completion=False)
comments_app = typer.Typer(no_args_is_help=True, add_completion=False)
blockers_app = typer.Typer(no_args_is_help=True, add_completion=False)
review_app = typer.Typer(no_args_is_help=True, add_completion=False)
auto_merge_app = typer.Typer(no_args_is_help=True, add_completion=False)
app.add_typer(account_app, name="account", help="Authenticated account operations")
app.add_typer(pr_app, name="pr", help="Pull request operations")
pr_app.add_typer(participants_app, name="participants", help="PR participants and reviewers")
pr_app.add_typer(comments_app, name="comments", help="PR comments")
pr_app.add_typer(blockers_app, name="blockers", help="PR blocker comments")
pr_app.add_typer(review_app, name="review", help="PR review workflow")
pr_app.add_typer(auto_merge_app, name="auto-merge", help="PR auto-merge")

batch_pr_app = typer.Typer(no_args_is_help=True, add_completion=False)
batch_pr_comments_app = typer.Typer(no_args_is_help=True, add_completion=False)
batch_pr_participants_app = typer.Typer(no_args_is_help=True, add_completion=False)
batch_pr_blockers_app = typer.Typer(no_args_is_help=True, add_completion=False)
batch_pr_review_app = typer.Typer(no_args_is_help=True, add_completion=False)
batch_pr_auto_merge_app = typer.Typer(no_args_is_help=True, add_completion=False)
pr_app.add_typer(batch_pr_app, name="batch", help="Batch pull request operations")
batch_pr_app.add_typer(batch_pr_comments_app, name="comments", help="Batch PR comment operations")
batch_pr_app.add_typer(batch_pr_participants_app, name="participants", help="Batch PR participant operations")
batch_pr_app.add_typer(batch_pr_blockers_app, name="blockers", help="Batch PR blocker comment operations")
batch_pr_app.add_typer(batch_pr_review_app, name="review", help="Batch PR review operations")
batch_pr_app.add_typer(batch_pr_auto_merge_app, name="auto-merge", help="Batch PR auto-merge operations")


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

    def paged_get_rest(
        self,
        path: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        limit: int = 50,
        max_items: int = 200,
    ) -> List[Dict[str, Any]]:
        """Fetch paged results from non-/api/latest REST namespaces."""
        out: List[Dict[str, Any]] = []
        start = 0
        params = dict(params or {})
        while True:
            params.update({"start": start, "limit": limit})
            page = self.request_rest("GET", path, params=params)
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


def _print_repositories(items: Iterable[Dict[str, Any]]) -> None:
    rows = []
    for repo in items:
        project = repo.get("project") or {}
        project_key = project.get("key") or ""
        slug = repo.get("slug") or ""
        name = repo.get("name") or ""
        rows.append((project_key, slug, name))

    if not rows:
        typer.echo("No repositories found.")
        return

    headers = ("PROJECT", "REPO", "NAME")
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


def _print_ssh_keys(items: Iterable[Dict[str, Any]]) -> None:
    rows = []
    for key in items:
        key_id = str(key.get("id", ""))
        label = key.get("label") or ""
        algorithm = key.get("algorithmType") or ""
        warning = key.get("warning") or ""
        rows.append((key_id, label, algorithm, warning))

    if not rows:
        typer.echo("No SSH keys found.")
        return

    headers = ("ID", "LABEL", "ALGO", "WARNING")
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


def _print_gpg_keys(items: Iterable[Dict[str, Any]]) -> None:
    rows = []
    for key in items:
        key_id = str(key.get("id", ""))
        email = key.get("emailAddress") or ""
        fingerprint = key.get("fingerprint") or ""
        rows.append((key_id, email, fingerprint))

    if not rows:
        typer.echo("No GPG keys found.")
        return

    headers = ("ID", "EMAIL", "FINGERPRINT")
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


def _resolve_user_slug(user_slug: Optional[str]) -> str:
    if user_slug:
        return user_slug.strip()
    for name in ("BITBUCKET_USER_SLUG", "BITBUCKET_USERNAME", "BITBUCKET_USER"):
        value = os.getenv(name, "").strip()
        if value:
            return value
    raise BBError(
        "Unable to resolve current user slug. Pass --user-slug or set BITBUCKET_USER_SLUG "
        "(or BITBUCKET_USERNAME / BITBUCKET_USER)."
    )


def _account_http_token_hint() -> str:
    return (
        "BBVA token note: Project/Repository HTTP access tokens may not have permission for "
        "user-account endpoints (ssh keys, gpg keys, user profile/settings). "
        "This is different from repository/project PR operations."
    )


def _format_account_error(error: BBError) -> str:
    message = str(error)
    if message.startswith("HTTP 401"):
        return f"{message}\nHint: {_account_http_token_hint()}"
    return message


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


def _load_json_value(source: str) -> Any:
    if source.startswith("@"):
        path = source[1:]
        if not path:
            raise BBError("Invalid --defaults value; '@' must be followed by a file path.")
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            raise BBError(f"Defaults file not found: {path}")
        except json.JSONDecodeError as e:
            raise BBError(f"Invalid JSON in defaults file {path}: {e}")
    try:
        return json.loads(source)
    except json.JSONDecodeError as e:
        raise BBError(f"Invalid JSON in --defaults: {e}")


def _load_batch_items(path: str) -> List[Dict[str, Any]]:
    try:
        if path == "-":
            data = json.load(sys.stdin)
        else:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
    except FileNotFoundError:
        raise BBError(f"Batch file not found: {path}")
    except json.JSONDecodeError as e:
        raise BBError(f"Invalid JSON in batch file {path}: {e}")

    if not isinstance(data, list):
        raise BBError("Batch file must contain a JSON list.")
    for i, item in enumerate(data, start=1):
        if not isinstance(item, dict):
            raise BBError(f"Batch item {i} must be an object.")
    return data


def _load_batch_defaults(
    defaults: Optional[str],
    project: Optional[str],
    repo: Optional[str],
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if defaults:
        data = _load_json_value(defaults)
        if not isinstance(data, dict):
            raise BBError("--defaults must be a JSON object.")
        out.update(data)
    if project:
        out["project"] = project
    if repo:
        out["repo"] = repo
    return out


def _merge_defaults(item: Dict[str, Any], defaults: Dict[str, Any]) -> Dict[str, Any]:
    if not defaults:
        return dict(item)
    merged = dict(defaults)
    merged.update(item)
    return merged


def _prepare_batch_items(
    file: str,
    defaults: Optional[str],
    project: Optional[str],
    repo: Optional[str],
) -> List[Dict[str, Any]]:
    items = _load_batch_items(file)
    defaults_map = _load_batch_defaults(defaults, project, repo)
    if not defaults_map:
        return items
    return [_merge_defaults(item, defaults_map) for item in items]


def _require_field(item: Dict[str, Any], field: str) -> Any:
    if field not in item or item[field] is None:
        raise BBError(f"Missing required field '{field}' in batch item.")
    return item[field]


def _coerce_str(value: Any, name: str) -> str:
    if isinstance(value, str):
        v = value.strip()
        if v:
            return v
    raise BBError(f"Invalid {name}; expected non-empty string.")


def _coerce_text(value: Any, name: str) -> str:
    if isinstance(value, str):
        return value
    raise BBError(f"Invalid {name}; expected string.")


def _coerce_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise BBError(f"Invalid {name}; expected integer.")
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        v = value.strip()
        try:
            return int(v)
        except ValueError:
            pass
    raise BBError(f"Invalid {name}; expected integer.")


def _coerce_bool(value: Any, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"true", "false"}:
            return v == "true"
    raise BBError(f"Invalid {name}; expected boolean.")


def _coerce_str_list(value: Any, name: str) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        v = value.strip()
        if not v:
            raise BBError(f"Invalid {name}; expected non-empty string or list of strings.")
        if "," in v:
            parts = [p.strip() for p in v.split(",") if p.strip()]
            if not parts:
                raise BBError(f"Invalid {name}; expected non-empty string or list of strings.")
            return parts
        return [v]
    if isinstance(value, list) and all(isinstance(x, str) for x in value):
        cleaned = [x.strip() for x in value if x.strip()]
        if not cleaned:
            raise BBError(f"Invalid {name}; expected non-empty list of strings.")
        return cleaned
    raise BBError(f"Invalid {name}; expected string or list of strings.")


def _optional_int(item: Dict[str, Any], field: str) -> Optional[int]:
    if field not in item or item[field] is None:
        return None
    return _coerce_int(item[field], field)


def _optional_text(item: Dict[str, Any], field: str) -> Optional[str]:
    if field not in item or item[field] is None:
        return None
    return _coerce_text(item[field], field)


def _optional_bool(item: Dict[str, Any], field: str) -> Optional[bool]:
    if field not in item or item[field] is None:
        return None
    return _coerce_bool(item[field], field)


def _optional_str(item: Dict[str, Any], field: str) -> Optional[str]:
    if field not in item or item[field] is None:
        return None
    return _coerce_str(item[field], field)


def _item_project_repo(item: Dict[str, Any]) -> tuple[str, str]:
    project = _coerce_str(_require_field(item, "project"), "project")
    repo = _coerce_str(_require_field(item, "repo"), "repo")
    return project, repo


def _item_pr(item: Dict[str, Any]) -> tuple[str, str, int]:
    project, repo = _item_project_repo(item)
    pr_id = _coerce_int(_require_field(item, "pr_id"), "pr_id")
    return project, repo, pr_id


def _item_comment(item: Dict[str, Any]) -> tuple[str, str, int, int]:
    project, repo, pr_id = _item_pr(item)
    comment_id = _coerce_int(_require_field(item, "comment_id"), "comment_id")
    return project, repo, pr_id, comment_id


def _item_reviewers(item: Dict[str, Any]) -> List[str]:
    if "reviewers" in item:
        return _coerce_str_list(item["reviewers"], "reviewers")
    if "reviewer" in item:
        return _coerce_str_list(item["reviewer"], "reviewer")
    return []


def _run_batch(
    items: List[Dict[str, Any]],
    op,
    *,
    concurrency: int,
    continue_on_error: bool,
) -> Dict[str, Any]:
    if concurrency < 1:
        raise BBError("--concurrency must be >= 1.")

    results: List[Dict[str, Any]] = []

    def run_one(idx: int, item: Dict[str, Any]) -> Dict[str, Any]:
        try:
            payload = op(item)
            if payload is None:
                payload = {}
            if not isinstance(payload, dict):
                payload = {"data": payload}
            return {"index": idx, "ok": True, "item": item, **payload}
        except Exception as e:
            return {"index": idx, "ok": False, "item": item, "error": str(e)}

    if concurrency == 1:
        for idx, item in enumerate(items, start=1):
            res = run_one(idx, item)
            results.append(res)
            if not res["ok"] and not continue_on_error:
                break
    else:
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = {
                executor.submit(run_one, idx, item): idx for idx, item in enumerate(items, start=1)
            }
            for future in as_completed(futures):
                res = future.result()
                results.append(res)
                if not res["ok"] and not continue_on_error:
                    for pending in futures:
                        if not pending.done():
                            pending.cancel()
                    break

    results.sort(key=lambda r: r["index"])
    ok_count = sum(1 for r in results if r.get("ok"))
    fail_count = sum(1 for r in results if not r.get("ok"))
    summary = {
        "total": len(items),
        "processed": len(results),
        "ok": ok_count,
        "failed": fail_count,
    }
    return {"summary": summary, "results": results}


def _print_batch(payload: Dict[str, Any], json_out: bool) -> None:
    if json_out:
        _print_json(payload)
        return
    results = payload.get("results", [])
    for r in results:
        index = r.get("index", "?")
        if r.get("ok"):
            msg = r.get("message") or "OK"
            typer.echo(f"[{index}] OK: {msg}")
        else:
            err = r.get("error") or "Unknown error"
            typer.echo(f"[{index}] ERROR: {err}")
    summary = payload.get("summary", {})
    total = summary.get("total", 0)
    processed = summary.get("processed", len(results))
    ok_count = summary.get("ok", 0)
    fail_count = summary.get("failed", 0)
    if processed != total:
        typer.echo(f"Summary: {ok_count} ok, {fail_count} failed (processed {processed}/{total}).")
    else:
        typer.echo(f"Summary: {ok_count} ok, {fail_count} failed.")


def _batch_execute(
    *,
    file: str,
    project: Optional[str],
    repo: Optional[str],
    defaults: Optional[str],
    concurrency: int,
    continue_on_error: bool,
    json_out: bool,
    op,
) -> None:
    bb = client()
    items = _prepare_batch_items(file, defaults, project, repo)
    payload = _run_batch(
        items,
        lambda item: op(bb, item),
        concurrency=concurrency,
        continue_on_error=continue_on_error,
    )
    _print_batch(payload, json_out)


def _op_pr_get(bb: BitbucketClient, project: str, repo: str, pr_id: int) -> Dict[str, Any]:
    path = f"projects/{project}/repos/{repo}/pull-requests/{pr_id}"
    pr = bb.request("GET", path)
    return {"message": f"Fetched PR #{pr_id}", "data": pr}


def _op_pr_create(
    bb: BitbucketClient,
    project: str,
    repo: str,
    from_branch: str,
    to_branch: str,
    title: str,
    description: str,
    reviewers: List[str],
    draft: Optional[bool],
) -> Dict[str, Any]:
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
    if reviewers:
        body["reviewers"] = [{"user": {"name": r}} for r in reviewers]
    if draft is not None:
        body["draft"] = bool(draft)
    path = f"projects/{project}/repos/{repo}/pull-requests"
    created = bb.request("POST", path, json_body=body)
    pr_id = created.get("id", "?")
    links = created.get("links", {}).get("self", [])
    url = links[0].get("href") if isinstance(links, list) and links else None
    message = f"Created PR #{pr_id}" + (f": {url}" if url else "")
    return {"message": message, "data": created}


def _op_pr_comment(bb: BitbucketClient, project: str, repo: str, pr_id: int, text: str) -> Dict[str, Any]:
    path = f"projects/{project}/repos/{repo}/pull-requests/{pr_id}/comments"
    body = {"text": text}
    resp = bb.request("POST", path, json_body=body)
    comment_id = resp.get("id", "?")
    return {"message": f"Added comment {comment_id} to PR #{pr_id}", "data": resp}


def _op_pr_approve(bb: BitbucketClient, project: str, repo: str, pr_id: int) -> Dict[str, Any]:
    path = f"projects/{project}/repos/{repo}/pull-requests/{pr_id}/approve"
    bb.request("POST", path)
    return {"message": f"Approved PR #{pr_id}"}


def _op_pr_unapprove(bb: BitbucketClient, project: str, repo: str, pr_id: int) -> Dict[str, Any]:
    path = f"projects/{project}/repos/{repo}/pull-requests/{pr_id}/approve"
    bb.request("DELETE", path)
    return {"message": f"Unapproved PR #{pr_id}"}


def _op_pr_decline(
    bb: BitbucketClient,
    project: str,
    repo: str,
    pr_id: int,
    version: Optional[int],
    comment: Optional[str],
) -> Dict[str, Any]:
    if version is None:
        version = _get_pr_version(bb, project, repo, pr_id)
    body: Dict[str, Any] = {"version": version}
    if comment:
        body["comment"] = comment
    path = f"projects/{project}/repos/{repo}/pull-requests/{pr_id}/decline"
    resp = bb.request("POST", path, params={"version": version}, json_body=body)
    return {"message": f"Declined PR #{pr_id}", "data": resp}


def _op_pr_reopen(
    bb: BitbucketClient, project: str, repo: str, pr_id: int, version: Optional[int]
) -> Dict[str, Any]:
    if version is None:
        version = _get_pr_version(bb, project, repo, pr_id)
    body = {"version": version}
    path = f"projects/{project}/repos/{repo}/pull-requests/{pr_id}/reopen"
    resp = bb.request("POST", path, params={"version": version}, json_body=body)
    return {"message": f"Re-opened PR #{pr_id}", "data": resp}


def _op_pr_merge_check(bb: BitbucketClient, project: str, repo: str, pr_id: int) -> Dict[str, Any]:
    path = f"projects/{project}/repos/{repo}/pull-requests/{pr_id}/merge"
    resp = bb.request("GET", path)
    return {"message": f"Merge check for PR #{pr_id}", "data": resp}


def _op_pr_merge(
    bb: BitbucketClient,
    project: str,
    repo: str,
    pr_id: int,
    version: Optional[int],
    message: Optional[str],
    strategy: Optional[str],
    auto_merge: Optional[bool],
    auto_subject: Optional[str],
) -> Dict[str, Any]:
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
    return {"message": f"Merged PR #{pr_id}", "data": resp}


def _op_pr_update(
    bb: BitbucketClient,
    project: str,
    repo: str,
    pr_id: int,
    version: Optional[int],
    title: Optional[str],
    description: Optional[str],
    reviewers: List[str],
    draft: Optional[bool],
) -> Dict[str, Any]:
    if title is None and description is None and not reviewers and draft is None:
        raise BBError("Nothing to update. Provide title, description, reviewers, or draft.")
    if version is None:
        version = _get_pr_version(bb, project, repo, pr_id)
    body: Dict[str, Any] = {"version": version}
    if title is not None:
        body["title"] = title
    if description is not None:
        body["description"] = description
    if reviewers:
        body["reviewers"] = [{"user": {"name": r}} for r in reviewers]
    if draft is not None:
        body["draft"] = bool(draft)
    path = f"projects/{project}/repos/{repo}/pull-requests/{pr_id}"
    resp = bb.request("PUT", path, json_body=body)
    new_version = resp.get("version", "?")
    return {"message": f"Updated PR #{pr_id} (version {new_version})", "data": resp}


def _op_pr_watch(bb: BitbucketClient, project: str, repo: str, pr_id: int) -> Dict[str, Any]:
    path = f"projects/{project}/repos/{repo}/pull-requests/{pr_id}/watch"
    bb.request("POST", path)
    return {"message": f"Watching PR #{pr_id}"}


def _op_pr_unwatch(bb: BitbucketClient, project: str, repo: str, pr_id: int) -> Dict[str, Any]:
    path = f"projects/{project}/repos/{repo}/pull-requests/{pr_id}/watch"
    bb.request("DELETE", path)
    return {"message": f"Stopped watching PR #{pr_id}"}


def _op_pr_merge_base(bb: BitbucketClient, project: str, repo: str, pr_id: int) -> Dict[str, Any]:
    path = f"projects/{project}/repos/{repo}/pull-requests/{pr_id}/merge-base"
    resp = bb.request("GET", path)
    return {"message": f"Merge base for PR #{pr_id}", "data": resp}


def _op_pr_commit_message(bb: BitbucketClient, project: str, repo: str, pr_id: int) -> Dict[str, Any]:
    path = f"projects/{project}/repos/{repo}/pull-requests/{pr_id}/commit-message-suggestion"
    resp = bb.request("GET", path)
    return {"message": f"Commit message suggestion for PR #{pr_id}", "data": resp}


def _op_pr_rebase_check(bb: BitbucketClient, project: str, repo: str, pr_id: int) -> Dict[str, Any]:
    path = f"git/latest/projects/{project}/repos/{repo}/pull-requests/{pr_id}/rebase"
    resp = bb.request_rest("GET", path)
    return {"message": f"Rebase check for PR #{pr_id}", "data": resp}


def _op_pr_rebase(
    bb: BitbucketClient, project: str, repo: str, pr_id: int, version: Optional[int]
) -> Dict[str, Any]:
    if version is None:
        version = _get_pr_version(bb, project, repo, pr_id)
    body = {"version": version}
    path = f"git/latest/projects/{project}/repos/{repo}/pull-requests/{pr_id}/rebase"
    resp = bb.request_rest("POST", path, json_body=body)
    return {"message": f"Rebased PR #{pr_id}", "data": resp}


def _op_pr_delete(
    bb: BitbucketClient, project: str, repo: str, pr_id: int, version: Optional[int]
) -> Dict[str, Any]:
    if version is None:
        version = _get_pr_version(bb, project, repo, pr_id)
    body = {"version": version}
    path = f"projects/{project}/repos/{repo}/pull-requests/{pr_id}"
    resp = bb.request("DELETE", path, json_body=body)
    return {"message": f"Deleted PR #{pr_id}", "data": resp}


def _op_pr_participants_add(
    bb: BitbucketClient, project: str, repo: str, pr_id: int, user: str, role: str
) -> Dict[str, Any]:
    role = _norm_choice(role, ROLE_CHOICES, "role")
    body = {"user": {"name": user}, "role": role}
    path = f"projects/{project}/repos/{repo}/pull-requests/{pr_id}/participants"
    resp = bb.request("POST", path, json_body=body)
    return {"message": f"Added {user} as {role} on PR #{pr_id}", "data": resp}


def _op_pr_participants_remove(
    bb: BitbucketClient, project: str, repo: str, pr_id: int, user: str
) -> Dict[str, Any]:
    path = f"projects/{project}/repos/{repo}/pull-requests/{pr_id}/participants/{user}"
    bb.request("DELETE", path)
    return {"message": f"Removed {user} from PR #{pr_id}"}


def _op_pr_participants_status(
    bb: BitbucketClient,
    project: str,
    repo: str,
    pr_id: int,
    user: str,
    status: str,
    last_reviewed_commit: Optional[str],
    version: Optional[int],
) -> Dict[str, Any]:
    status = _norm_choice(status, STATUS_CHOICES, "status")
    body: Dict[str, Any] = {"status": status}
    if last_reviewed_commit:
        body["lastReviewedCommit"] = last_reviewed_commit
    params = {"version": version} if version is not None else None
    path = f"projects/{project}/repos/{repo}/pull-requests/{pr_id}/participants/{user}"
    resp = bb.request("PUT", path, params=params, json_body=body)
    return {"message": f"Updated {user} status to {status} on PR #{pr_id}", "data": resp}


def _op_pr_comments_get(
    bb: BitbucketClient, project: str, repo: str, pr_id: int, comment_id: int
) -> Dict[str, Any]:
    path = f"projects/{project}/repos/{repo}/pull-requests/{pr_id}/comments/{comment_id}"
    resp = bb.request("GET", path)
    return {"message": f"Fetched comment {comment_id} on PR #{pr_id}", "data": resp}


def _op_pr_comments_update(
    bb: BitbucketClient,
    project: str,
    repo: str,
    pr_id: int,
    comment_id: int,
    text: Optional[str],
    severity: Optional[str],
    state: Optional[str],
    version: Optional[int],
) -> Dict[str, Any]:
    if text is None and severity is None and state is None:
        raise BBError("Nothing to update. Provide text, severity, or state.")
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
    return {"message": f"Updated comment {comment_id} on PR #{pr_id}", "data": resp}


def _op_pr_comments_delete(
    bb: BitbucketClient,
    project: str,
    repo: str,
    pr_id: int,
    comment_id: int,
    version: Optional[int],
) -> Dict[str, Any]:
    if version is None:
        version = _get_comment_version(bb, project, repo, pr_id, comment_id)
    path = f"projects/{project}/repos/{repo}/pull-requests/{pr_id}/comments/{comment_id}"
    bb.request("DELETE", path, params={"version": version})
    return {"message": f"Deleted comment {comment_id} on PR #{pr_id}"}


def _op_pr_comments_apply_suggestion(
    bb: BitbucketClient,
    project: str,
    repo: str,
    pr_id: int,
    comment_id: int,
    suggestion_index: int,
    comment_version: Optional[int],
    pr_version: Optional[int],
    commit_message: Optional[str],
) -> Dict[str, Any]:
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
    return {"message": f"Applied suggestion from comment {comment_id} on PR #{pr_id}", "data": resp}


def _op_pr_comments_react(
    bb: BitbucketClient, project: str, repo: str, pr_id: int, comment_id: int, emoticon: str
) -> Dict[str, Any]:
    path = (
        f"comment-likes/latest/projects/{project}/repos/{repo}/pull-requests/{pr_id}"
        f"/comments/{comment_id}/reactions/{emoticon}"
    )
    bb.request_rest("PUT", path)
    return {"message": f"Reacted to comment {comment_id} on PR #{pr_id}"}


def _op_pr_comments_unreact(
    bb: BitbucketClient, project: str, repo: str, pr_id: int, comment_id: int, emoticon: str
) -> Dict[str, Any]:
    path = (
        f"comment-likes/latest/projects/{project}/repos/{repo}/pull-requests/{pr_id}"
        f"/comments/{comment_id}/reactions/{emoticon}"
    )
    bb.request_rest("DELETE", path)
    return {"message": f"Removed reaction from comment {comment_id} on PR #{pr_id}"}


def _op_pr_blockers_add(
    bb: BitbucketClient, project: str, repo: str, pr_id: int, text: str
) -> Dict[str, Any]:
    path = f"projects/{project}/repos/{repo}/pull-requests/{pr_id}/blocker-comments"
    body = {"text": text}
    resp = bb.request("POST", path, json_body=body)
    comment_id = resp.get("id", "?")
    return {"message": f"Added blocker comment {comment_id} to PR #{pr_id}", "data": resp}


def _op_pr_blockers_get(
    bb: BitbucketClient, project: str, repo: str, pr_id: int, comment_id: int
) -> Dict[str, Any]:
    path = f"projects/{project}/repos/{repo}/pull-requests/{pr_id}/blocker-comments/{comment_id}"
    resp = bb.request("GET", path)
    return {"message": f"Fetched blocker comment {comment_id} on PR #{pr_id}", "data": resp}


def _op_pr_blockers_update(
    bb: BitbucketClient,
    project: str,
    repo: str,
    pr_id: int,
    comment_id: int,
    text: Optional[str],
    severity: Optional[str],
    state: Optional[str],
    version: Optional[int],
) -> Dict[str, Any]:
    if text is None and severity is None and state is None:
        raise BBError("Nothing to update. Provide text, severity, or state.")
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
    return {"message": f"Updated blocker comment {comment_id} on PR #{pr_id}", "data": resp}


def _op_pr_blockers_delete(
    bb: BitbucketClient,
    project: str,
    repo: str,
    pr_id: int,
    comment_id: int,
    version: Optional[int],
) -> Dict[str, Any]:
    if version is None:
        version = _get_comment_version(bb, project, repo, pr_id, comment_id, blocker=True)
    path = f"projects/{project}/repos/{repo}/pull-requests/{pr_id}/blocker-comments/{comment_id}"
    bb.request("DELETE", path, params={"version": version})
    return {"message": f"Deleted blocker comment {comment_id} on PR #{pr_id}"}


def _op_pr_review_get(bb: BitbucketClient, project: str, repo: str, pr_id: int) -> Dict[str, Any]:
    path = f"projects/{project}/repos/{repo}/pull-requests/{pr_id}/review"
    resp = bb.request("GET", path)
    return {"message": f"Fetched review for PR #{pr_id}", "data": resp}


def _op_pr_review_complete(
    bb: BitbucketClient,
    project: str,
    repo: str,
    pr_id: int,
    comment_text: Optional[str],
    last_reviewed_commit: Optional[str],
    status: Optional[str],
) -> Dict[str, Any]:
    if comment_text is None and last_reviewed_commit is None and status is None:
        raise BBError("Nothing to update. Provide comment, last-reviewed-commit, or status.")
    body: Dict[str, Any] = {}
    if comment_text is not None:
        body["commentText"] = comment_text
    if last_reviewed_commit is not None:
        body["lastReviewedCommit"] = last_reviewed_commit
    if status is not None:
        body["participantStatus"] = _norm_choice(status, STATUS_CHOICES, "status")
    path = f"projects/{project}/repos/{repo}/pull-requests/{pr_id}/review"
    resp = bb.request("PUT", path, json_body=body)
    return {"message": f"Completed review for PR #{pr_id}", "data": resp}


def _op_pr_review_discard(
    bb: BitbucketClient, project: str, repo: str, pr_id: int
) -> Dict[str, Any]:
    path = f"projects/{project}/repos/{repo}/pull-requests/{pr_id}/review"
    bb.request("DELETE", path)
    return {"message": f"Discarded review for PR #{pr_id}"}


def _op_pr_auto_merge_get(
    bb: BitbucketClient, project: str, repo: str, pr_id: int
) -> Dict[str, Any]:
    path = f"projects/{project}/repos/{repo}/pull-requests/{pr_id}/auto-merge"
    resp = bb.request("GET", path)
    return {"message": f"Fetched auto-merge for PR #{pr_id}", "data": resp}


def _op_pr_auto_merge_set(
    bb: BitbucketClient, project: str, repo: str, pr_id: int
) -> Dict[str, Any]:
    path = f"projects/{project}/repos/{repo}/pull-requests/{pr_id}/auto-merge"
    bb.request("POST", path)
    return {"message": f"Requested auto-merge for PR #{pr_id}"}


def _op_pr_auto_merge_cancel(
    bb: BitbucketClient, project: str, repo: str, pr_id: int
) -> Dict[str, Any]:
    path = f"projects/{project}/repos/{repo}/pull-requests/{pr_id}/auto-merge"
    bb.request("DELETE", path)
    return {"message": f"Cancelled auto-merge for PR #{pr_id}"}


def _op_account_recent_repos(
    bb: BitbucketClient, limit: int, max_items: int
) -> Dict[str, Any]:
    repos = bb.paged_get("profile/recent/repos", limit=limit, max_items=max_items)
    return {"message": f"Fetched {len(repos)} recently accessed repositories", "data": repos}


def _op_account_ssh_keys(
    bb: BitbucketClient, user: Optional[str], limit: int, max_items: int
) -> Dict[str, Any]:
    params = {"user": user} if user else None
    keys = bb.paged_get_rest("ssh/latest/keys", params=params, limit=limit, max_items=max_items)
    who = user or "current user"
    return {"message": f"Fetched {len(keys)} SSH keys for {who}", "data": keys}


def _op_account_gpg_keys(
    bb: BitbucketClient, user: Optional[str], limit: int, max_items: int
) -> Dict[str, Any]:
    params = {"user": user} if user else None
    keys = bb.paged_get_rest("gpg/latest/keys", params=params, limit=limit, max_items=max_items)
    who = user or "current user"
    return {"message": f"Fetched {len(keys)} GPG keys for {who}", "data": keys}


def _op_account_user(bb: BitbucketClient, user_slug: str) -> Dict[str, Any]:
    path = f"users/{user_slug}"
    user = bb.request("GET", path)
    return {"message": f"Fetched user '{user_slug}'", "data": user}


def _op_account_user_settings(bb: BitbucketClient, user_slug: str) -> Dict[str, Any]:
    path = f"users/{user_slug}/settings"
    settings = bb.request("GET", path)
    return {"message": f"Fetched settings for user '{user_slug}'", "data": settings}


def _op_account_me(
    bb: BitbucketClient,
    *,
    user_slug: Optional[str],
    include_profile: bool,
    include_settings: bool,
    limit: int,
    max_items: int,
) -> Dict[str, Any]:
    errors: Dict[str, str] = {}

    def collect_list(
        key: str,
        getter,
    ) -> List[Dict[str, Any]]:
        try:
            return getter()["data"]
        except BBError as e:
            errors[key] = _format_account_error(e)
            return []

    recent_repos = collect_list(
        "recent_repositories",
        lambda: _op_account_recent_repos(bb, limit=limit, max_items=max_items),
    )
    ssh_keys = collect_list(
        "ssh_keys",
        lambda: _op_account_ssh_keys(bb, user=None, limit=limit, max_items=max_items),
    )
    gpg_keys = collect_list(
        "gpg_keys",
        lambda: _op_account_gpg_keys(bb, user=None, limit=limit, max_items=max_items),
    )

    resolved_user_slug: Optional[str] = None
    user_profile: Optional[Dict[str, Any]] = None
    user_settings: Optional[Dict[str, Any]] = None
    if include_profile or include_settings:
        try:
            resolved_user_slug = _resolve_user_slug(user_slug)
        except BBError as e:
            errors["resolved_user_slug"] = _format_account_error(e)
            resolved_user_slug = None

        if resolved_user_slug:
            if include_profile:
                try:
                    user_profile = _op_account_user(bb, resolved_user_slug)["data"]
                except BBError as e:
                    errors["user"] = _format_account_error(e)
            if include_settings:
                try:
                    user_settings = _op_account_user_settings(bb, resolved_user_slug)["data"]
                except BBError as e:
                    errors["settings"] = _format_account_error(e)

    payload: Dict[str, Any] = {
        "counts": {
            "recent_repos": len(recent_repos),
            "ssh_keys": len(ssh_keys),
            "gpg_keys": len(gpg_keys),
        },
        "recent_repositories": recent_repos,
        "ssh_keys": ssh_keys,
        "gpg_keys": gpg_keys,
    }
    if resolved_user_slug:
        payload["resolved_user_slug"] = resolved_user_slug
    if user_profile is not None:
        payload["user"] = user_profile
    if user_settings is not None:
        payload["settings"] = user_settings
    if errors:
        payload["partial"] = True
        payload["errors"] = errors
        payload["notes"] = [
            "Some account endpoints were not accessible with the current token.",
            _account_http_token_hint(),
        ]
        return {
            "message": f"Fetched partial authenticated account information ({len(errors)} section(s) failed)",
            "data": payload,
        }

    return {"message": "Fetched authenticated account information", "data": payload}


def _batch_op_pr_get(bb: BitbucketClient, item: Dict[str, Any]) -> Dict[str, Any]:
    project, repo, pr_id = _item_pr(item)
    return _op_pr_get(bb, project, repo, pr_id)


def _batch_op_pr_create(bb: BitbucketClient, item: Dict[str, Any]) -> Dict[str, Any]:
    project, repo = _item_project_repo(item)
    from_branch = _coerce_str(_require_field(item, "from_branch"), "from_branch")
    to_branch = _coerce_str(_require_field(item, "to_branch"), "to_branch")
    title = _coerce_str(_require_field(item, "title"), "title")
    description = _optional_text(item, "description")
    if description is None:
        description = ""
    reviewers = _item_reviewers(item)
    draft = _optional_bool(item, "draft")
    return _op_pr_create(
        bb,
        project,
        repo,
        from_branch,
        to_branch,
        title,
        description,
        reviewers,
        draft,
    )


def _batch_op_pr_comment(bb: BitbucketClient, item: Dict[str, Any]) -> Dict[str, Any]:
    project, repo, pr_id = _item_pr(item)
    text = _coerce_str(_require_field(item, "text"), "text")
    return _op_pr_comment(bb, project, repo, pr_id, text)


def _batch_op_pr_approve(bb: BitbucketClient, item: Dict[str, Any]) -> Dict[str, Any]:
    project, repo, pr_id = _item_pr(item)
    return _op_pr_approve(bb, project, repo, pr_id)


def _batch_op_pr_unapprove(bb: BitbucketClient, item: Dict[str, Any]) -> Dict[str, Any]:
    project, repo, pr_id = _item_pr(item)
    return _op_pr_unapprove(bb, project, repo, pr_id)


def _batch_op_pr_decline(bb: BitbucketClient, item: Dict[str, Any]) -> Dict[str, Any]:
    project, repo, pr_id = _item_pr(item)
    version = _optional_int(item, "version")
    comment = _optional_text(item, "comment")
    return _op_pr_decline(bb, project, repo, pr_id, version, comment)


def _batch_op_pr_reopen(bb: BitbucketClient, item: Dict[str, Any]) -> Dict[str, Any]:
    project, repo, pr_id = _item_pr(item)
    version = _optional_int(item, "version")
    return _op_pr_reopen(bb, project, repo, pr_id, version)


def _batch_op_pr_merge_check(bb: BitbucketClient, item: Dict[str, Any]) -> Dict[str, Any]:
    project, repo, pr_id = _item_pr(item)
    return _op_pr_merge_check(bb, project, repo, pr_id)


def _batch_op_pr_merge(bb: BitbucketClient, item: Dict[str, Any]) -> Dict[str, Any]:
    project, repo, pr_id = _item_pr(item)
    version = _optional_int(item, "version")
    message = _optional_text(item, "message")
    strategy = _optional_text(item, "strategy")
    auto_merge = _optional_bool(item, "auto_merge")
    auto_subject = _optional_text(item, "auto_subject")
    return _op_pr_merge(
        bb,
        project,
        repo,
        pr_id,
        version,
        message,
        strategy,
        auto_merge,
        auto_subject,
    )


def _batch_op_pr_update(bb: BitbucketClient, item: Dict[str, Any]) -> Dict[str, Any]:
    project, repo, pr_id = _item_pr(item)
    version = _optional_int(item, "version")
    title = _optional_text(item, "title")
    description = _optional_text(item, "description")
    reviewers = _item_reviewers(item)
    draft = _optional_bool(item, "draft")
    return _op_pr_update(bb, project, repo, pr_id, version, title, description, reviewers, draft)


def _batch_op_pr_watch(bb: BitbucketClient, item: Dict[str, Any]) -> Dict[str, Any]:
    project, repo, pr_id = _item_pr(item)
    return _op_pr_watch(bb, project, repo, pr_id)


def _batch_op_pr_unwatch(bb: BitbucketClient, item: Dict[str, Any]) -> Dict[str, Any]:
    project, repo, pr_id = _item_pr(item)
    return _op_pr_unwatch(bb, project, repo, pr_id)


def _batch_op_pr_merge_base(bb: BitbucketClient, item: Dict[str, Any]) -> Dict[str, Any]:
    project, repo, pr_id = _item_pr(item)
    return _op_pr_merge_base(bb, project, repo, pr_id)


def _batch_op_pr_commit_message(bb: BitbucketClient, item: Dict[str, Any]) -> Dict[str, Any]:
    project, repo, pr_id = _item_pr(item)
    return _op_pr_commit_message(bb, project, repo, pr_id)


def _batch_op_pr_rebase_check(bb: BitbucketClient, item: Dict[str, Any]) -> Dict[str, Any]:
    project, repo, pr_id = _item_pr(item)
    return _op_pr_rebase_check(bb, project, repo, pr_id)


def _batch_op_pr_rebase(bb: BitbucketClient, item: Dict[str, Any]) -> Dict[str, Any]:
    project, repo, pr_id = _item_pr(item)
    version = _optional_int(item, "version")
    return _op_pr_rebase(bb, project, repo, pr_id, version)


def _batch_op_pr_delete(bb: BitbucketClient, item: Dict[str, Any]) -> Dict[str, Any]:
    project, repo, pr_id = _item_pr(item)
    version = _optional_int(item, "version")
    return _op_pr_delete(bb, project, repo, pr_id, version)


def _batch_op_pr_participants_add(bb: BitbucketClient, item: Dict[str, Any]) -> Dict[str, Any]:
    project, repo, pr_id = _item_pr(item)
    user = _coerce_str(_require_field(item, "user"), "user")
    role = _coerce_str(item.get("role", "REVIEWER"), "role")
    return _op_pr_participants_add(bb, project, repo, pr_id, user, role)


def _batch_op_pr_participants_remove(bb: BitbucketClient, item: Dict[str, Any]) -> Dict[str, Any]:
    project, repo, pr_id = _item_pr(item)
    user = _coerce_str(_require_field(item, "user"), "user")
    return _op_pr_participants_remove(bb, project, repo, pr_id, user)


def _batch_op_pr_participants_status(bb: BitbucketClient, item: Dict[str, Any]) -> Dict[str, Any]:
    project, repo, pr_id = _item_pr(item)
    user = _coerce_str(_require_field(item, "user"), "user")
    status = _coerce_str(_require_field(item, "status"), "status")
    last_reviewed_commit = _optional_text(item, "last_reviewed_commit")
    version = _optional_int(item, "version")
    return _op_pr_participants_status(
        bb, project, repo, pr_id, user, status, last_reviewed_commit, version
    )


def _batch_op_pr_comments_add(bb: BitbucketClient, item: Dict[str, Any]) -> Dict[str, Any]:
    return _batch_op_pr_comment(bb, item)


def _batch_op_pr_comments_get(bb: BitbucketClient, item: Dict[str, Any]) -> Dict[str, Any]:
    project, repo, pr_id, comment_id = _item_comment(item)
    return _op_pr_comments_get(bb, project, repo, pr_id, comment_id)


def _batch_op_pr_comments_update(bb: BitbucketClient, item: Dict[str, Any]) -> Dict[str, Any]:
    project, repo, pr_id, comment_id = _item_comment(item)
    text = _optional_text(item, "text")
    severity = _optional_text(item, "severity")
    state = _optional_text(item, "state")
    version = _optional_int(item, "version")
    return _op_pr_comments_update(bb, project, repo, pr_id, comment_id, text, severity, state, version)


def _batch_op_pr_comments_delete(bb: BitbucketClient, item: Dict[str, Any]) -> Dict[str, Any]:
    project, repo, pr_id, comment_id = _item_comment(item)
    version = _optional_int(item, "version")
    return _op_pr_comments_delete(bb, project, repo, pr_id, comment_id, version)


def _batch_op_pr_comments_apply_suggestion(bb: BitbucketClient, item: Dict[str, Any]) -> Dict[str, Any]:
    project, repo, pr_id, comment_id = _item_comment(item)
    suggestion_index = _coerce_int(_require_field(item, "suggestion_index"), "suggestion_index")
    comment_version = _optional_int(item, "comment_version")
    pr_version = _optional_int(item, "pr_version")
    commit_message = _optional_text(item, "commit_message")
    return _op_pr_comments_apply_suggestion(
        bb,
        project,
        repo,
        pr_id,
        comment_id,
        suggestion_index,
        comment_version,
        pr_version,
        commit_message,
    )


def _batch_op_pr_comments_react(bb: BitbucketClient, item: Dict[str, Any]) -> Dict[str, Any]:
    project, repo, pr_id, comment_id = _item_comment(item)
    emoticon = _coerce_str(_require_field(item, "emoticon"), "emoticon")
    return _op_pr_comments_react(bb, project, repo, pr_id, comment_id, emoticon)


def _batch_op_pr_comments_unreact(bb: BitbucketClient, item: Dict[str, Any]) -> Dict[str, Any]:
    project, repo, pr_id, comment_id = _item_comment(item)
    emoticon = _coerce_str(_require_field(item, "emoticon"), "emoticon")
    return _op_pr_comments_unreact(bb, project, repo, pr_id, comment_id, emoticon)


def _batch_op_pr_blockers_add(bb: BitbucketClient, item: Dict[str, Any]) -> Dict[str, Any]:
    project, repo, pr_id = _item_pr(item)
    text = _coerce_str(_require_field(item, "text"), "text")
    return _op_pr_blockers_add(bb, project, repo, pr_id, text)


def _batch_op_pr_blockers_get(bb: BitbucketClient, item: Dict[str, Any]) -> Dict[str, Any]:
    project, repo, pr_id, comment_id = _item_comment(item)
    return _op_pr_blockers_get(bb, project, repo, pr_id, comment_id)


def _batch_op_pr_blockers_update(bb: BitbucketClient, item: Dict[str, Any]) -> Dict[str, Any]:
    project, repo, pr_id, comment_id = _item_comment(item)
    text = _optional_text(item, "text")
    severity = _optional_text(item, "severity")
    state = _optional_text(item, "state")
    version = _optional_int(item, "version")
    return _op_pr_blockers_update(
        bb, project, repo, pr_id, comment_id, text, severity, state, version
    )


def _batch_op_pr_blockers_delete(bb: BitbucketClient, item: Dict[str, Any]) -> Dict[str, Any]:
    project, repo, pr_id, comment_id = _item_comment(item)
    version = _optional_int(item, "version")
    return _op_pr_blockers_delete(bb, project, repo, pr_id, comment_id, version)


def _batch_op_pr_review_get(bb: BitbucketClient, item: Dict[str, Any]) -> Dict[str, Any]:
    project, repo, pr_id = _item_pr(item)
    return _op_pr_review_get(bb, project, repo, pr_id)


def _batch_op_pr_review_complete(bb: BitbucketClient, item: Dict[str, Any]) -> Dict[str, Any]:
    project, repo, pr_id = _item_pr(item)
    comment_text = _optional_text(item, "comment")
    last_reviewed_commit = _optional_text(item, "last_reviewed_commit")
    status = _optional_text(item, "status")
    return _op_pr_review_complete(
        bb, project, repo, pr_id, comment_text, last_reviewed_commit, status
    )


def _batch_op_pr_review_discard(bb: BitbucketClient, item: Dict[str, Any]) -> Dict[str, Any]:
    project, repo, pr_id = _item_pr(item)
    return _op_pr_review_discard(bb, project, repo, pr_id)


def _batch_op_pr_auto_merge_get(bb: BitbucketClient, item: Dict[str, Any]) -> Dict[str, Any]:
    project, repo, pr_id = _item_pr(item)
    return _op_pr_auto_merge_get(bb, project, repo, pr_id)


def _batch_op_pr_auto_merge_set(bb: BitbucketClient, item: Dict[str, Any]) -> Dict[str, Any]:
    project, repo, pr_id = _item_pr(item)
    return _op_pr_auto_merge_set(bb, project, repo, pr_id)


def _batch_op_pr_auto_merge_cancel(bb: BitbucketClient, item: Dict[str, Any]) -> Dict[str, Any]:
    project, repo, pr_id = _item_pr(item)
    return _op_pr_auto_merge_cancel(bb, project, repo, pr_id)


@account_app.command("recent-repos")
def account_recent_repos(
    limit: int = typer.Option(50, help="Page size"),
    max_items: int = typer.Option(200, help="Max items to fetch across pages"),
    json_out: bool = typer.Option(False, "--json", help="Print raw JSON response"),
):
    """Get recently accessed repositories for the authenticated user."""
    bb = client()
    try:
        resp = _op_account_recent_repos(bb, limit=limit, max_items=max_items)
    except BBError as e:
        raise BBError(_format_account_error(e))
    if json_out:
        _print_json(resp["data"])
    else:
        _print_repositories(resp["data"])


@account_app.command("ssh-keys")
def account_ssh_keys(
    user: Optional[str] = typer.Option(None, "--user", help="Optional user slug/name; defaults to current user"),
    limit: int = typer.Option(50, help="Page size"),
    max_items: int = typer.Option(200, help="Max items to fetch across pages"),
    json_out: bool = typer.Option(False, "--json", help="Print raw JSON response"),
):
    """Get SSH keys for the authenticated user (or another user with sufficient permissions)."""
    bb = client()
    try:
        resp = _op_account_ssh_keys(bb, user=user, limit=limit, max_items=max_items)
    except BBError as e:
        raise BBError(_format_account_error(e))
    if json_out:
        _print_json(resp["data"])
    else:
        _print_ssh_keys(resp["data"])


@account_app.command("gpg-keys")
def account_gpg_keys(
    user: Optional[str] = typer.Option(None, "--user", help="Optional user slug/name; defaults to current user"),
    limit: int = typer.Option(50, help="Page size"),
    max_items: int = typer.Option(200, help="Max items to fetch across pages"),
    json_out: bool = typer.Option(False, "--json", help="Print raw JSON response"),
):
    """Get GPG keys for the authenticated user (or another user with sufficient permissions)."""
    bb = client()
    try:
        resp = _op_account_gpg_keys(bb, user=user, limit=limit, max_items=max_items)
    except BBError as e:
        raise BBError(_format_account_error(e))
    if json_out:
        _print_json(resp["data"])
    else:
        _print_gpg_keys(resp["data"])


@account_app.command("user")
def account_user(
    user_slug: Optional[str] = typer.Option(
        None,
        "--user-slug",
        help="User slug. If omitted, resolves from BITBUCKET_USER_SLUG / BITBUCKET_USERNAME / BITBUCKET_USER.",
    ),
):
    """Get user details for the authenticated user account (or a supplied user slug)."""
    bb = client()
    try:
        slug = _resolve_user_slug(user_slug)
        resp = _op_account_user(bb, slug)
    except BBError as e:
        raise BBError(_format_account_error(e))
    _print_json(resp["data"])


@account_app.command("settings")
def account_settings(
    user_slug: Optional[str] = typer.Option(
        None,
        "--user-slug",
        help="User slug. If omitted, resolves from BITBUCKET_USER_SLUG / BITBUCKET_USERNAME / BITBUCKET_USER.",
    ),
):
    """Get account settings for the authenticated user account (or a supplied user slug)."""
    bb = client()
    try:
        slug = _resolve_user_slug(user_slug)
        resp = _op_account_user_settings(bb, slug)
    except BBError as e:
        raise BBError(_format_account_error(e))
    _print_json(resp["data"])


@account_app.command("me")
def account_me(
    user_slug: Optional[str] = typer.Option(
        None,
        "--user-slug",
        help="Optional user slug used when profile/settings are requested.",
    ),
    include_profile: bool = typer.Option(
        True,
        "--include-profile/--no-include-profile",
        help="Include /api/latest/users/{userSlug} details.",
    ),
    include_settings: bool = typer.Option(
        False,
        "--include-settings/--no-include-settings",
        help="Include /api/latest/users/{userSlug}/settings.",
    ),
    limit: int = typer.Option(25, help="Page size for account-related paged endpoints"),
    max_items: int = typer.Option(100, help="Max items per account-related endpoint"),
):
    """Get a consolidated snapshot of the authenticated account.

    This command returns partial data when some account endpoints are unauthorized for the token in use.
    """
    bb = client()
    resp = _op_account_me(
        bb,
        user_slug=user_slug,
        include_profile=include_profile,
        include_settings=include_settings,
        limit=limit,
        max_items=max_items,
    )
    _print_json(resp["data"])


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
    resp = _op_pr_get(bb, project, repo, pr_id)
    _print_json(resp["data"])


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
    created = _op_pr_create(
        bb,
        project,
        repo,
        from_branch,
        to_branch,
        title,
        description,
        reviewer,
        draft,
    )
    if json_out:
        _print_json(created["data"])
    else:
        typer.echo(created["message"])


@pr_app.command("comment")
def pr_comment(
    project: str = typer.Option(..., "--project", "-p"),
    repo: str = typer.Option(..., "--repo", "-r"),
    pr_id: int = typer.Argument(..., help="Pull request numeric ID"),
    text: str = typer.Option(..., "--text", "-t", help="Comment text"),
):
    """Add a comment to a pull request."""
    bb = client()
    resp = _op_pr_comment(bb, project, repo, pr_id, text)
    typer.echo(resp["message"])


@pr_app.command("approve")
def pr_approve(
    project: str = typer.Option(..., "--project", "-p"),
    repo: str = typer.Option(..., "--repo", "-r"),
    pr_id: int = typer.Argument(..., help="Pull request numeric ID"),
):
    """Approve a pull request."""
    bb = client()
    resp = _op_pr_approve(bb, project, repo, pr_id)
    typer.echo(resp["message"])


@pr_app.command("unapprove")
def pr_unapprove(
    project: str = typer.Option(..., "--project", "-p"),
    repo: str = typer.Option(..., "--repo", "-r"),
    pr_id: int = typer.Argument(..., help="Pull request numeric ID"),
):
    """Unapprove a pull request."""
    bb = client()
    resp = _op_pr_unapprove(bb, project, repo, pr_id)
    typer.echo(resp["message"])


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
    resp = _op_pr_decline(bb, project, repo, pr_id, version, comment or None)
    if json_out:
        _print_json(resp["data"])
    else:
        typer.echo(resp["message"])


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
    resp = _op_pr_reopen(bb, project, repo, pr_id, version)
    if json_out:
        _print_json(resp["data"])
    else:
        typer.echo(resp["message"])


@pr_app.command("merge-check")
def pr_merge_check(
    project: str = typer.Option(..., "--project", "-p"),
    repo: str = typer.Option(..., "--repo", "-r"),
    pr_id: int = typer.Argument(..., help="Pull request numeric ID"),
):
    """Test if a pull request can be merged."""
    bb = client()
    resp = _op_pr_merge_check(bb, project, repo, pr_id)
    _print_json(resp["data"])


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
    resp = _op_pr_merge(
        bb,
        project,
        repo,
        pr_id,
        version,
        message or None,
        strategy or None,
        auto_merge,
        auto_subject or None,
    )
    if json_out:
        _print_json(resp["data"])
    else:
        typer.echo(resp["message"])


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
    bb = client()
    resp = _op_pr_update(bb, project, repo, pr_id, version, title, description, reviewer, draft)
    if json_out:
        _print_json(resp["data"])
    else:
        typer.echo(resp["message"])


@pr_app.command("watch")
def pr_watch(
    project: str = typer.Option(..., "--project", "-p"),
    repo: str = typer.Option(..., "--repo", "-r"),
    pr_id: int = typer.Argument(..., help="Pull request numeric ID"),
):
    """Watch a pull request."""
    bb = client()
    resp = _op_pr_watch(bb, project, repo, pr_id)
    typer.echo(resp["message"])


@pr_app.command("unwatch")
def pr_unwatch(
    project: str = typer.Option(..., "--project", "-p"),
    repo: str = typer.Option(..., "--repo", "-r"),
    pr_id: int = typer.Argument(..., help="Pull request numeric ID"),
):
    """Stop watching a pull request."""
    bb = client()
    resp = _op_pr_unwatch(bb, project, repo, pr_id)
    typer.echo(resp["message"])


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
    resp = _op_pr_participants_add(bb, project, repo, pr_id, user, role)
    if json_out:
        _print_json(resp["data"])
    else:
        typer.echo(resp["message"])


@participants_app.command("remove")
def pr_participants_remove(
    project: str = typer.Option(..., "--project", "-p"),
    repo: str = typer.Option(..., "--repo", "-r"),
    pr_id: int = typer.Argument(..., help="Pull request numeric ID"),
    user: str = typer.Argument(..., help="User slug to remove"),
):
    """Remove a participant/reviewer from a pull request."""
    bb = client()
    resp = _op_pr_participants_remove(bb, project, repo, pr_id, user)
    typer.echo(resp["message"])


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
    resp = _op_pr_participants_status(
        bb, project, repo, pr_id, user, status, last_reviewed_commit, version
    )
    if json_out:
        _print_json(resp["data"])
    else:
        typer.echo(resp["message"])


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
    resp = _op_pr_comments_get(bb, project, repo, pr_id, comment_id)
    _print_json(resp["data"])


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
    bb = client()
    resp = _op_pr_comments_update(
        bb, project, repo, pr_id, comment_id, text, severity, state, version
    )
    if json_out:
        _print_json(resp["data"])
    else:
        typer.echo(resp["message"])


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
    resp = _op_pr_comments_delete(bb, project, repo, pr_id, comment_id, version)
    typer.echo(resp["message"])


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
    resp = _op_pr_comments_apply_suggestion(
        bb,
        project,
        repo,
        pr_id,
        comment_id,
        suggestion_index,
        comment_version,
        pr_version,
        commit_message or None,
    )
    if json_out:
        _print_json(resp["data"])
    else:
        typer.echo(resp["message"])


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
    resp = _op_pr_comments_react(bb, project, repo, pr_id, comment_id, emoticon)
    typer.echo(resp["message"])


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
    resp = _op_pr_comments_unreact(bb, project, repo, pr_id, comment_id, emoticon)
    typer.echo(resp["message"])


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
    resp = _op_pr_blockers_add(bb, project, repo, pr_id, text)
    if json_out:
        _print_json(resp["data"])
    else:
        typer.echo(resp["message"])


@blockers_app.command("get")
def pr_blockers_get(
    project: str = typer.Option(..., "--project", "-p"),
    repo: str = typer.Option(..., "--repo", "-r"),
    pr_id: int = typer.Argument(..., help="Pull request numeric ID"),
    comment_id: int = typer.Argument(..., help="Comment ID"),
):
    """Get a blocker comment."""
    bb = client()
    resp = _op_pr_blockers_get(bb, project, repo, pr_id, comment_id)
    _print_json(resp["data"])


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
    bb = client()
    resp = _op_pr_blockers_update(
        bb, project, repo, pr_id, comment_id, text, severity, state, version
    )
    if json_out:
        _print_json(resp["data"])
    else:
        typer.echo(resp["message"])


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
    resp = _op_pr_blockers_delete(bb, project, repo, pr_id, comment_id, version)
    typer.echo(resp["message"])


@review_app.command("get")
def pr_review_get(
    project: str = typer.Option(..., "--project", "-p"),
    repo: str = typer.Option(..., "--repo", "-r"),
    pr_id: int = typer.Argument(..., help="Pull request numeric ID"),
):
    """Get pull request review thread."""
    bb = client()
    resp = _op_pr_review_get(bb, project, repo, pr_id)
    _print_json(resp["data"])


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
    bb = client()
    resp = _op_pr_review_complete(
        bb, project, repo, pr_id, comment_text, last_reviewed_commit, status
    )
    if json_out:
        _print_json(resp["data"])
    else:
        typer.echo(resp["message"])


@review_app.command("discard")
def pr_review_discard(
    project: str = typer.Option(..., "--project", "-p"),
    repo: str = typer.Option(..., "--repo", "-r"),
    pr_id: int = typer.Argument(..., help="Pull request numeric ID"),
):
    """Discard pull request review."""
    bb = client()
    resp = _op_pr_review_discard(bb, project, repo, pr_id)
    typer.echo(resp["message"])


@auto_merge_app.command("get")
def pr_auto_merge_get(
    project: str = typer.Option(..., "--project", "-p"),
    repo: str = typer.Option(..., "--repo", "-r"),
    pr_id: int = typer.Argument(..., help="Pull request numeric ID"),
):
    """Get auto-merge request for a pull request."""
    bb = client()
    resp = _op_pr_auto_merge_get(bb, project, repo, pr_id)
    _print_json(resp["data"])


@auto_merge_app.command("set")
def pr_auto_merge_set(
    project: str = typer.Option(..., "--project", "-p"),
    repo: str = typer.Option(..., "--repo", "-r"),
    pr_id: int = typer.Argument(..., help="Pull request numeric ID"),
):
    """Request auto-merge for a pull request."""
    bb = client()
    resp = _op_pr_auto_merge_set(bb, project, repo, pr_id)
    typer.echo(resp["message"])


@auto_merge_app.command("cancel")
def pr_auto_merge_cancel(
    project: str = typer.Option(..., "--project", "-p"),
    repo: str = typer.Option(..., "--repo", "-r"),
    pr_id: int = typer.Argument(..., help="Pull request numeric ID"),
):
    """Cancel auto-merge for a pull request."""
    bb = client()
    resp = _op_pr_auto_merge_cancel(bb, project, repo, pr_id)
    typer.echo(resp["message"])


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
    resp = _op_pr_merge_base(bb, project, repo, pr_id)
    _print_json(resp["data"])


@pr_app.command("commit-message")
def pr_commit_message(
    project: str = typer.Option(..., "--project", "-p"),
    repo: str = typer.Option(..., "--repo", "-r"),
    pr_id: int = typer.Argument(..., help="Pull request numeric ID"),
):
    """Get commit message suggestion for a pull request."""
    bb = client()
    resp = _op_pr_commit_message(bb, project, repo, pr_id)
    _print_json(resp["data"])


@pr_app.command("rebase-check")
def pr_rebase_check(
    project: str = typer.Option(..., "--project", "-p"),
    repo: str = typer.Option(..., "--repo", "-r"),
    pr_id: int = typer.Argument(..., help="Pull request numeric ID"),
):
    """Check whether a pull request can be rebased."""
    bb = client()
    resp = _op_pr_rebase_check(bb, project, repo, pr_id)
    _print_json(resp["data"])


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
    resp = _op_pr_rebase(bb, project, repo, pr_id, version)
    if json_out:
        _print_json(resp["data"])
    else:
        typer.echo(resp["message"])


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
    resp = _op_pr_delete(bb, project, repo, pr_id, version)
    if json_out:
        _print_json(resp["data"])
    else:
        typer.echo(resp["message"])


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


@batch_pr_app.command("get")
def pr_batch_get(
    file: str = typer.Option(..., "--file", "-f", help="Path to JSON list (or '-' for stdin)"),
    project: Optional[str] = typer.Option(None, "--project", "-p", help="Default project key"),
    repo: Optional[str] = typer.Option(None, "--repo", "-r", help="Default repo slug"),
    defaults: Optional[str] = typer.Option(None, "--defaults", help="JSON object or @file with default fields"),
    concurrency: int = typer.Option(1, "--concurrency", "-c", help="Number of concurrent requests"),
    continue_on_error: bool = typer.Option(True, "--continue-on-error/--stop-on-error"),
    json_out: bool = typer.Option(False, "--json", help="Print raw JSON results"),
):
    """Batch get pull requests."""
    _batch_execute(
        file=file,
        project=project,
        repo=repo,
        defaults=defaults,
        concurrency=concurrency,
        continue_on_error=continue_on_error,
        json_out=json_out,
        op=_batch_op_pr_get,
    )


@batch_pr_app.command("create")
def pr_batch_create(
    file: str = typer.Option(..., "--file", "-f", help="Path to JSON list (or '-' for stdin)"),
    project: Optional[str] = typer.Option(None, "--project", "-p", help="Default project key"),
    repo: Optional[str] = typer.Option(None, "--repo", "-r", help="Default repo slug"),
    defaults: Optional[str] = typer.Option(None, "--defaults", help="JSON object or @file with default fields"),
    concurrency: int = typer.Option(1, "--concurrency", "-c", help="Number of concurrent requests"),
    continue_on_error: bool = typer.Option(True, "--continue-on-error/--stop-on-error"),
    json_out: bool = typer.Option(False, "--json", help="Print raw JSON results"),
):
    """Batch create pull requests."""
    _batch_execute(
        file=file,
        project=project,
        repo=repo,
        defaults=defaults,
        concurrency=concurrency,
        continue_on_error=continue_on_error,
        json_out=json_out,
        op=_batch_op_pr_create,
    )


@batch_pr_app.command("comment")
def pr_batch_comment(
    file: str = typer.Option(..., "--file", "-f", help="Path to JSON list (or '-' for stdin)"),
    project: Optional[str] = typer.Option(None, "--project", "-p", help="Default project key"),
    repo: Optional[str] = typer.Option(None, "--repo", "-r", help="Default repo slug"),
    defaults: Optional[str] = typer.Option(None, "--defaults", help="JSON object or @file with default fields"),
    concurrency: int = typer.Option(1, "--concurrency", "-c", help="Number of concurrent requests"),
    continue_on_error: bool = typer.Option(True, "--continue-on-error/--stop-on-error"),
    json_out: bool = typer.Option(False, "--json", help="Print raw JSON results"),
):
    """Batch add comments to pull requests."""
    _batch_execute(
        file=file,
        project=project,
        repo=repo,
        defaults=defaults,
        concurrency=concurrency,
        continue_on_error=continue_on_error,
        json_out=json_out,
        op=_batch_op_pr_comment,
    )


@batch_pr_app.command("approve")
def pr_batch_approve(
    file: str = typer.Option(..., "--file", "-f", help="Path to JSON list (or '-' for stdin)"),
    project: Optional[str] = typer.Option(None, "--project", "-p", help="Default project key"),
    repo: Optional[str] = typer.Option(None, "--repo", "-r", help="Default repo slug"),
    defaults: Optional[str] = typer.Option(None, "--defaults", help="JSON object or @file with default fields"),
    concurrency: int = typer.Option(1, "--concurrency", "-c", help="Number of concurrent requests"),
    continue_on_error: bool = typer.Option(True, "--continue-on-error/--stop-on-error"),
    json_out: bool = typer.Option(False, "--json", help="Print raw JSON results"),
):
    """Batch approve pull requests."""
    _batch_execute(
        file=file,
        project=project,
        repo=repo,
        defaults=defaults,
        concurrency=concurrency,
        continue_on_error=continue_on_error,
        json_out=json_out,
        op=_batch_op_pr_approve,
    )


@batch_pr_app.command("unapprove")
def pr_batch_unapprove(
    file: str = typer.Option(..., "--file", "-f", help="Path to JSON list (or '-' for stdin)"),
    project: Optional[str] = typer.Option(None, "--project", "-p", help="Default project key"),
    repo: Optional[str] = typer.Option(None, "--repo", "-r", help="Default repo slug"),
    defaults: Optional[str] = typer.Option(None, "--defaults", help="JSON object or @file with default fields"),
    concurrency: int = typer.Option(1, "--concurrency", "-c", help="Number of concurrent requests"),
    continue_on_error: bool = typer.Option(True, "--continue-on-error/--stop-on-error"),
    json_out: bool = typer.Option(False, "--json", help="Print raw JSON results"),
):
    """Batch unapprove pull requests."""
    _batch_execute(
        file=file,
        project=project,
        repo=repo,
        defaults=defaults,
        concurrency=concurrency,
        continue_on_error=continue_on_error,
        json_out=json_out,
        op=_batch_op_pr_unapprove,
    )


@batch_pr_app.command("decline")
def pr_batch_decline(
    file: str = typer.Option(..., "--file", "-f", help="Path to JSON list (or '-' for stdin)"),
    project: Optional[str] = typer.Option(None, "--project", "-p", help="Default project key"),
    repo: Optional[str] = typer.Option(None, "--repo", "-r", help="Default repo slug"),
    defaults: Optional[str] = typer.Option(None, "--defaults", help="JSON object or @file with default fields"),
    concurrency: int = typer.Option(1, "--concurrency", "-c", help="Number of concurrent requests"),
    continue_on_error: bool = typer.Option(True, "--continue-on-error/--stop-on-error"),
    json_out: bool = typer.Option(False, "--json", help="Print raw JSON results"),
):
    """Batch decline pull requests."""
    _batch_execute(
        file=file,
        project=project,
        repo=repo,
        defaults=defaults,
        concurrency=concurrency,
        continue_on_error=continue_on_error,
        json_out=json_out,
        op=_batch_op_pr_decline,
    )


@batch_pr_app.command("reopen")
def pr_batch_reopen(
    file: str = typer.Option(..., "--file", "-f", help="Path to JSON list (or '-' for stdin)"),
    project: Optional[str] = typer.Option(None, "--project", "-p", help="Default project key"),
    repo: Optional[str] = typer.Option(None, "--repo", "-r", help="Default repo slug"),
    defaults: Optional[str] = typer.Option(None, "--defaults", help="JSON object or @file with default fields"),
    concurrency: int = typer.Option(1, "--concurrency", "-c", help="Number of concurrent requests"),
    continue_on_error: bool = typer.Option(True, "--continue-on-error/--stop-on-error"),
    json_out: bool = typer.Option(False, "--json", help="Print raw JSON results"),
):
    """Batch reopen pull requests."""
    _batch_execute(
        file=file,
        project=project,
        repo=repo,
        defaults=defaults,
        concurrency=concurrency,
        continue_on_error=continue_on_error,
        json_out=json_out,
        op=_batch_op_pr_reopen,
    )


@batch_pr_app.command("merge-check")
def pr_batch_merge_check(
    file: str = typer.Option(..., "--file", "-f", help="Path to JSON list (or '-' for stdin)"),
    project: Optional[str] = typer.Option(None, "--project", "-p", help="Default project key"),
    repo: Optional[str] = typer.Option(None, "--repo", "-r", help="Default repo slug"),
    defaults: Optional[str] = typer.Option(None, "--defaults", help="JSON object or @file with default fields"),
    concurrency: int = typer.Option(1, "--concurrency", "-c", help="Number of concurrent requests"),
    continue_on_error: bool = typer.Option(True, "--continue-on-error/--stop-on-error"),
    json_out: bool = typer.Option(False, "--json", help="Print raw JSON results"),
):
    """Batch merge-check pull requests."""
    _batch_execute(
        file=file,
        project=project,
        repo=repo,
        defaults=defaults,
        concurrency=concurrency,
        continue_on_error=continue_on_error,
        json_out=json_out,
        op=_batch_op_pr_merge_check,
    )


@batch_pr_app.command("merge")
def pr_batch_merge(
    file: str = typer.Option(..., "--file", "-f", help="Path to JSON list (or '-' for stdin)"),
    project: Optional[str] = typer.Option(None, "--project", "-p", help="Default project key"),
    repo: Optional[str] = typer.Option(None, "--repo", "-r", help="Default repo slug"),
    defaults: Optional[str] = typer.Option(None, "--defaults", help="JSON object or @file with default fields"),
    concurrency: int = typer.Option(1, "--concurrency", "-c", help="Number of concurrent requests"),
    continue_on_error: bool = typer.Option(True, "--continue-on-error/--stop-on-error"),
    json_out: bool = typer.Option(False, "--json", help="Print raw JSON results"),
):
    """Batch merge pull requests."""
    _batch_execute(
        file=file,
        project=project,
        repo=repo,
        defaults=defaults,
        concurrency=concurrency,
        continue_on_error=continue_on_error,
        json_out=json_out,
        op=_batch_op_pr_merge,
    )


@batch_pr_app.command("update")
def pr_batch_update(
    file: str = typer.Option(..., "--file", "-f", help="Path to JSON list (or '-' for stdin)"),
    project: Optional[str] = typer.Option(None, "--project", "-p", help="Default project key"),
    repo: Optional[str] = typer.Option(None, "--repo", "-r", help="Default repo slug"),
    defaults: Optional[str] = typer.Option(None, "--defaults", help="JSON object or @file with default fields"),
    concurrency: int = typer.Option(1, "--concurrency", "-c", help="Number of concurrent requests"),
    continue_on_error: bool = typer.Option(True, "--continue-on-error/--stop-on-error"),
    json_out: bool = typer.Option(False, "--json", help="Print raw JSON results"),
):
    """Batch update pull requests."""
    _batch_execute(
        file=file,
        project=project,
        repo=repo,
        defaults=defaults,
        concurrency=concurrency,
        continue_on_error=continue_on_error,
        json_out=json_out,
        op=_batch_op_pr_update,
    )


@batch_pr_app.command("watch")
def pr_batch_watch(
    file: str = typer.Option(..., "--file", "-f", help="Path to JSON list (or '-' for stdin)"),
    project: Optional[str] = typer.Option(None, "--project", "-p", help="Default project key"),
    repo: Optional[str] = typer.Option(None, "--repo", "-r", help="Default repo slug"),
    defaults: Optional[str] = typer.Option(None, "--defaults", help="JSON object or @file with default fields"),
    concurrency: int = typer.Option(1, "--concurrency", "-c", help="Number of concurrent requests"),
    continue_on_error: bool = typer.Option(True, "--continue-on-error/--stop-on-error"),
    json_out: bool = typer.Option(False, "--json", help="Print raw JSON results"),
):
    """Batch watch pull requests."""
    _batch_execute(
        file=file,
        project=project,
        repo=repo,
        defaults=defaults,
        concurrency=concurrency,
        continue_on_error=continue_on_error,
        json_out=json_out,
        op=_batch_op_pr_watch,
    )


@batch_pr_app.command("unwatch")
def pr_batch_unwatch(
    file: str = typer.Option(..., "--file", "-f", help="Path to JSON list (or '-' for stdin)"),
    project: Optional[str] = typer.Option(None, "--project", "-p", help="Default project key"),
    repo: Optional[str] = typer.Option(None, "--repo", "-r", help="Default repo slug"),
    defaults: Optional[str] = typer.Option(None, "--defaults", help="JSON object or @file with default fields"),
    concurrency: int = typer.Option(1, "--concurrency", "-c", help="Number of concurrent requests"),
    continue_on_error: bool = typer.Option(True, "--continue-on-error/--stop-on-error"),
    json_out: bool = typer.Option(False, "--json", help="Print raw JSON results"),
):
    """Batch unwatch pull requests."""
    _batch_execute(
        file=file,
        project=project,
        repo=repo,
        defaults=defaults,
        concurrency=concurrency,
        continue_on_error=continue_on_error,
        json_out=json_out,
        op=_batch_op_pr_unwatch,
    )


@batch_pr_app.command("merge-base")
def pr_batch_merge_base(
    file: str = typer.Option(..., "--file", "-f", help="Path to JSON list (or '-' for stdin)"),
    project: Optional[str] = typer.Option(None, "--project", "-p", help="Default project key"),
    repo: Optional[str] = typer.Option(None, "--repo", "-r", help="Default repo slug"),
    defaults: Optional[str] = typer.Option(None, "--defaults", help="JSON object or @file with default fields"),
    concurrency: int = typer.Option(1, "--concurrency", "-c", help="Number of concurrent requests"),
    continue_on_error: bool = typer.Option(True, "--continue-on-error/--stop-on-error"),
    json_out: bool = typer.Option(False, "--json", help="Print raw JSON results"),
):
    """Batch get merge bases for pull requests."""
    _batch_execute(
        file=file,
        project=project,
        repo=repo,
        defaults=defaults,
        concurrency=concurrency,
        continue_on_error=continue_on_error,
        json_out=json_out,
        op=_batch_op_pr_merge_base,
    )


@batch_pr_app.command("commit-message")
def pr_batch_commit_message(
    file: str = typer.Option(..., "--file", "-f", help="Path to JSON list (or '-' for stdin)"),
    project: Optional[str] = typer.Option(None, "--project", "-p", help="Default project key"),
    repo: Optional[str] = typer.Option(None, "--repo", "-r", help="Default repo slug"),
    defaults: Optional[str] = typer.Option(None, "--defaults", help="JSON object or @file with default fields"),
    concurrency: int = typer.Option(1, "--concurrency", "-c", help="Number of concurrent requests"),
    continue_on_error: bool = typer.Option(True, "--continue-on-error/--stop-on-error"),
    json_out: bool = typer.Option(False, "--json", help="Print raw JSON results"),
):
    """Batch get commit message suggestions for pull requests."""
    _batch_execute(
        file=file,
        project=project,
        repo=repo,
        defaults=defaults,
        concurrency=concurrency,
        continue_on_error=continue_on_error,
        json_out=json_out,
        op=_batch_op_pr_commit_message,
    )


@batch_pr_app.command("rebase-check")
def pr_batch_rebase_check(
    file: str = typer.Option(..., "--file", "-f", help="Path to JSON list (or '-' for stdin)"),
    project: Optional[str] = typer.Option(None, "--project", "-p", help="Default project key"),
    repo: Optional[str] = typer.Option(None, "--repo", "-r", help="Default repo slug"),
    defaults: Optional[str] = typer.Option(None, "--defaults", help="JSON object or @file with default fields"),
    concurrency: int = typer.Option(1, "--concurrency", "-c", help="Number of concurrent requests"),
    continue_on_error: bool = typer.Option(True, "--continue-on-error/--stop-on-error"),
    json_out: bool = typer.Option(False, "--json", help="Print raw JSON results"),
):
    """Batch rebase-check pull requests."""
    _batch_execute(
        file=file,
        project=project,
        repo=repo,
        defaults=defaults,
        concurrency=concurrency,
        continue_on_error=continue_on_error,
        json_out=json_out,
        op=_batch_op_pr_rebase_check,
    )


@batch_pr_app.command("rebase")
def pr_batch_rebase(
    file: str = typer.Option(..., "--file", "-f", help="Path to JSON list (or '-' for stdin)"),
    project: Optional[str] = typer.Option(None, "--project", "-p", help="Default project key"),
    repo: Optional[str] = typer.Option(None, "--repo", "-r", help="Default repo slug"),
    defaults: Optional[str] = typer.Option(None, "--defaults", help="JSON object or @file with default fields"),
    concurrency: int = typer.Option(1, "--concurrency", "-c", help="Number of concurrent requests"),
    continue_on_error: bool = typer.Option(True, "--continue-on-error/--stop-on-error"),
    json_out: bool = typer.Option(False, "--json", help="Print raw JSON results"),
):
    """Batch rebase pull requests."""
    _batch_execute(
        file=file,
        project=project,
        repo=repo,
        defaults=defaults,
        concurrency=concurrency,
        continue_on_error=continue_on_error,
        json_out=json_out,
        op=_batch_op_pr_rebase,
    )


@batch_pr_app.command("delete")
def pr_batch_delete(
    file: str = typer.Option(..., "--file", "-f", help="Path to JSON list (or '-' for stdin)"),
    project: Optional[str] = typer.Option(None, "--project", "-p", help="Default project key"),
    repo: Optional[str] = typer.Option(None, "--repo", "-r", help="Default repo slug"),
    defaults: Optional[str] = typer.Option(None, "--defaults", help="JSON object or @file with default fields"),
    concurrency: int = typer.Option(1, "--concurrency", "-c", help="Number of concurrent requests"),
    continue_on_error: bool = typer.Option(True, "--continue-on-error/--stop-on-error"),
    json_out: bool = typer.Option(False, "--json", help="Print raw JSON results"),
):
    """Batch delete pull requests."""
    _batch_execute(
        file=file,
        project=project,
        repo=repo,
        defaults=defaults,
        concurrency=concurrency,
        continue_on_error=continue_on_error,
        json_out=json_out,
        op=_batch_op_pr_delete,
    )


@batch_pr_participants_app.command("add")
def pr_batch_participants_add(
    file: str = typer.Option(..., "--file", "-f", help="Path to JSON list (or '-' for stdin)"),
    project: Optional[str] = typer.Option(None, "--project", "-p", help="Default project key"),
    repo: Optional[str] = typer.Option(None, "--repo", "-r", help="Default repo slug"),
    defaults: Optional[str] = typer.Option(None, "--defaults", help="JSON object or @file with default fields"),
    concurrency: int = typer.Option(1, "--concurrency", "-c", help="Number of concurrent requests"),
    continue_on_error: bool = typer.Option(True, "--continue-on-error/--stop-on-error"),
    json_out: bool = typer.Option(False, "--json", help="Print raw JSON results"),
):
    """Batch add participants to pull requests."""
    _batch_execute(
        file=file,
        project=project,
        repo=repo,
        defaults=defaults,
        concurrency=concurrency,
        continue_on_error=continue_on_error,
        json_out=json_out,
        op=_batch_op_pr_participants_add,
    )


@batch_pr_participants_app.command("remove")
def pr_batch_participants_remove(
    file: str = typer.Option(..., "--file", "-f", help="Path to JSON list (or '-' for stdin)"),
    project: Optional[str] = typer.Option(None, "--project", "-p", help="Default project key"),
    repo: Optional[str] = typer.Option(None, "--repo", "-r", help="Default repo slug"),
    defaults: Optional[str] = typer.Option(None, "--defaults", help="JSON object or @file with default fields"),
    concurrency: int = typer.Option(1, "--concurrency", "-c", help="Number of concurrent requests"),
    continue_on_error: bool = typer.Option(True, "--continue-on-error/--stop-on-error"),
    json_out: bool = typer.Option(False, "--json", help="Print raw JSON results"),
):
    """Batch remove participants from pull requests."""
    _batch_execute(
        file=file,
        project=project,
        repo=repo,
        defaults=defaults,
        concurrency=concurrency,
        continue_on_error=continue_on_error,
        json_out=json_out,
        op=_batch_op_pr_participants_remove,
    )


@batch_pr_participants_app.command("status")
def pr_batch_participants_status(
    file: str = typer.Option(..., "--file", "-f", help="Path to JSON list (or '-' for stdin)"),
    project: Optional[str] = typer.Option(None, "--project", "-p", help="Default project key"),
    repo: Optional[str] = typer.Option(None, "--repo", "-r", help="Default repo slug"),
    defaults: Optional[str] = typer.Option(None, "--defaults", help="JSON object or @file with default fields"),
    concurrency: int = typer.Option(1, "--concurrency", "-c", help="Number of concurrent requests"),
    continue_on_error: bool = typer.Option(True, "--continue-on-error/--stop-on-error"),
    json_out: bool = typer.Option(False, "--json", help="Print raw JSON results"),
):
    """Batch update participant status on pull requests."""
    _batch_execute(
        file=file,
        project=project,
        repo=repo,
        defaults=defaults,
        concurrency=concurrency,
        continue_on_error=continue_on_error,
        json_out=json_out,
        op=_batch_op_pr_participants_status,
    )


@batch_pr_comments_app.command("add")
def pr_batch_comments_add(
    file: str = typer.Option(..., "--file", "-f", help="Path to JSON list (or '-' for stdin)"),
    project: Optional[str] = typer.Option(None, "--project", "-p", help="Default project key"),
    repo: Optional[str] = typer.Option(None, "--repo", "-r", help="Default repo slug"),
    defaults: Optional[str] = typer.Option(None, "--defaults", help="JSON object or @file with default fields"),
    concurrency: int = typer.Option(1, "--concurrency", "-c", help="Number of concurrent requests"),
    continue_on_error: bool = typer.Option(True, "--continue-on-error/--stop-on-error"),
    json_out: bool = typer.Option(False, "--json", help="Print raw JSON results"),
):
    """Batch add comments to pull requests."""
    _batch_execute(
        file=file,
        project=project,
        repo=repo,
        defaults=defaults,
        concurrency=concurrency,
        continue_on_error=continue_on_error,
        json_out=json_out,
        op=_batch_op_pr_comments_add,
    )


@batch_pr_comments_app.command("get")
def pr_batch_comments_get(
    file: str = typer.Option(..., "--file", "-f", help="Path to JSON list (or '-' for stdin)"),
    project: Optional[str] = typer.Option(None, "--project", "-p", help="Default project key"),
    repo: Optional[str] = typer.Option(None, "--repo", "-r", help="Default repo slug"),
    defaults: Optional[str] = typer.Option(None, "--defaults", help="JSON object or @file with default fields"),
    concurrency: int = typer.Option(1, "--concurrency", "-c", help="Number of concurrent requests"),
    continue_on_error: bool = typer.Option(True, "--continue-on-error/--stop-on-error"),
    json_out: bool = typer.Option(False, "--json", help="Print raw JSON results"),
):
    """Batch get pull request comments."""
    _batch_execute(
        file=file,
        project=project,
        repo=repo,
        defaults=defaults,
        concurrency=concurrency,
        continue_on_error=continue_on_error,
        json_out=json_out,
        op=_batch_op_pr_comments_get,
    )


@batch_pr_comments_app.command("update")
def pr_batch_comments_update(
    file: str = typer.Option(..., "--file", "-f", help="Path to JSON list (or '-' for stdin)"),
    project: Optional[str] = typer.Option(None, "--project", "-p", help="Default project key"),
    repo: Optional[str] = typer.Option(None, "--repo", "-r", help="Default repo slug"),
    defaults: Optional[str] = typer.Option(None, "--defaults", help="JSON object or @file with default fields"),
    concurrency: int = typer.Option(1, "--concurrency", "-c", help="Number of concurrent requests"),
    continue_on_error: bool = typer.Option(True, "--continue-on-error/--stop-on-error"),
    json_out: bool = typer.Option(False, "--json", help="Print raw JSON results"),
):
    """Batch update pull request comments."""
    _batch_execute(
        file=file,
        project=project,
        repo=repo,
        defaults=defaults,
        concurrency=concurrency,
        continue_on_error=continue_on_error,
        json_out=json_out,
        op=_batch_op_pr_comments_update,
    )


@batch_pr_comments_app.command("delete")
def pr_batch_comments_delete(
    file: str = typer.Option(..., "--file", "-f", help="Path to JSON list (or '-' for stdin)"),
    project: Optional[str] = typer.Option(None, "--project", "-p", help="Default project key"),
    repo: Optional[str] = typer.Option(None, "--repo", "-r", help="Default repo slug"),
    defaults: Optional[str] = typer.Option(None, "--defaults", help="JSON object or @file with default fields"),
    concurrency: int = typer.Option(1, "--concurrency", "-c", help="Number of concurrent requests"),
    continue_on_error: bool = typer.Option(True, "--continue-on-error/--stop-on-error"),
    json_out: bool = typer.Option(False, "--json", help="Print raw JSON results"),
):
    """Batch delete pull request comments."""
    _batch_execute(
        file=file,
        project=project,
        repo=repo,
        defaults=defaults,
        concurrency=concurrency,
        continue_on_error=continue_on_error,
        json_out=json_out,
        op=_batch_op_pr_comments_delete,
    )


@batch_pr_comments_app.command("apply-suggestion")
def pr_batch_comments_apply_suggestion(
    file: str = typer.Option(..., "--file", "-f", help="Path to JSON list (or '-' for stdin)"),
    project: Optional[str] = typer.Option(None, "--project", "-p", help="Default project key"),
    repo: Optional[str] = typer.Option(None, "--repo", "-r", help="Default repo slug"),
    defaults: Optional[str] = typer.Option(None, "--defaults", help="JSON object or @file with default fields"),
    concurrency: int = typer.Option(1, "--concurrency", "-c", help="Number of concurrent requests"),
    continue_on_error: bool = typer.Option(True, "--continue-on-error/--stop-on-error"),
    json_out: bool = typer.Option(False, "--json", help="Print raw JSON results"),
):
    """Batch apply suggestions from pull request comments."""
    _batch_execute(
        file=file,
        project=project,
        repo=repo,
        defaults=defaults,
        concurrency=concurrency,
        continue_on_error=continue_on_error,
        json_out=json_out,
        op=_batch_op_pr_comments_apply_suggestion,
    )


@batch_pr_comments_app.command("react")
def pr_batch_comments_react(
    file: str = typer.Option(..., "--file", "-f", help="Path to JSON list (or '-' for stdin)"),
    project: Optional[str] = typer.Option(None, "--project", "-p", help="Default project key"),
    repo: Optional[str] = typer.Option(None, "--repo", "-r", help="Default repo slug"),
    defaults: Optional[str] = typer.Option(None, "--defaults", help="JSON object or @file with default fields"),
    concurrency: int = typer.Option(1, "--concurrency", "-c", help="Number of concurrent requests"),
    continue_on_error: bool = typer.Option(True, "--continue-on-error/--stop-on-error"),
    json_out: bool = typer.Option(False, "--json", help="Print raw JSON results"),
):
    """Batch react to pull request comments."""
    _batch_execute(
        file=file,
        project=project,
        repo=repo,
        defaults=defaults,
        concurrency=concurrency,
        continue_on_error=continue_on_error,
        json_out=json_out,
        op=_batch_op_pr_comments_react,
    )


@batch_pr_comments_app.command("unreact")
def pr_batch_comments_unreact(
    file: str = typer.Option(..., "--file", "-f", help="Path to JSON list (or '-' for stdin)"),
    project: Optional[str] = typer.Option(None, "--project", "-p", help="Default project key"),
    repo: Optional[str] = typer.Option(None, "--repo", "-r", help="Default repo slug"),
    defaults: Optional[str] = typer.Option(None, "--defaults", help="JSON object or @file with default fields"),
    concurrency: int = typer.Option(1, "--concurrency", "-c", help="Number of concurrent requests"),
    continue_on_error: bool = typer.Option(True, "--continue-on-error/--stop-on-error"),
    json_out: bool = typer.Option(False, "--json", help="Print raw JSON results"),
):
    """Batch remove reactions from pull request comments."""
    _batch_execute(
        file=file,
        project=project,
        repo=repo,
        defaults=defaults,
        concurrency=concurrency,
        continue_on_error=continue_on_error,
        json_out=json_out,
        op=_batch_op_pr_comments_unreact,
    )


@batch_pr_blockers_app.command("add")
def pr_batch_blockers_add(
    file: str = typer.Option(..., "--file", "-f", help="Path to JSON list (or '-' for stdin)"),
    project: Optional[str] = typer.Option(None, "--project", "-p", help="Default project key"),
    repo: Optional[str] = typer.Option(None, "--repo", "-r", help="Default repo slug"),
    defaults: Optional[str] = typer.Option(None, "--defaults", help="JSON object or @file with default fields"),
    concurrency: int = typer.Option(1, "--concurrency", "-c", help="Number of concurrent requests"),
    continue_on_error: bool = typer.Option(True, "--continue-on-error/--stop-on-error"),
    json_out: bool = typer.Option(False, "--json", help="Print raw JSON results"),
):
    """Batch add blocker comments to pull requests."""
    _batch_execute(
        file=file,
        project=project,
        repo=repo,
        defaults=defaults,
        concurrency=concurrency,
        continue_on_error=continue_on_error,
        json_out=json_out,
        op=_batch_op_pr_blockers_add,
    )


@batch_pr_blockers_app.command("get")
def pr_batch_blockers_get(
    file: str = typer.Option(..., "--file", "-f", help="Path to JSON list (or '-' for stdin)"),
    project: Optional[str] = typer.Option(None, "--project", "-p", help="Default project key"),
    repo: Optional[str] = typer.Option(None, "--repo", "-r", help="Default repo slug"),
    defaults: Optional[str] = typer.Option(None, "--defaults", help="JSON object or @file with default fields"),
    concurrency: int = typer.Option(1, "--concurrency", "-c", help="Number of concurrent requests"),
    continue_on_error: bool = typer.Option(True, "--continue-on-error/--stop-on-error"),
    json_out: bool = typer.Option(False, "--json", help="Print raw JSON results"),
):
    """Batch get blocker comments."""
    _batch_execute(
        file=file,
        project=project,
        repo=repo,
        defaults=defaults,
        concurrency=concurrency,
        continue_on_error=continue_on_error,
        json_out=json_out,
        op=_batch_op_pr_blockers_get,
    )


@batch_pr_blockers_app.command("update")
def pr_batch_blockers_update(
    file: str = typer.Option(..., "--file", "-f", help="Path to JSON list (or '-' for stdin)"),
    project: Optional[str] = typer.Option(None, "--project", "-p", help="Default project key"),
    repo: Optional[str] = typer.Option(None, "--repo", "-r", help="Default repo slug"),
    defaults: Optional[str] = typer.Option(None, "--defaults", help="JSON object or @file with default fields"),
    concurrency: int = typer.Option(1, "--concurrency", "-c", help="Number of concurrent requests"),
    continue_on_error: bool = typer.Option(True, "--continue-on-error/--stop-on-error"),
    json_out: bool = typer.Option(False, "--json", help="Print raw JSON results"),
):
    """Batch update blocker comments."""
    _batch_execute(
        file=file,
        project=project,
        repo=repo,
        defaults=defaults,
        concurrency=concurrency,
        continue_on_error=continue_on_error,
        json_out=json_out,
        op=_batch_op_pr_blockers_update,
    )


@batch_pr_blockers_app.command("delete")
def pr_batch_blockers_delete(
    file: str = typer.Option(..., "--file", "-f", help="Path to JSON list (or '-' for stdin)"),
    project: Optional[str] = typer.Option(None, "--project", "-p", help="Default project key"),
    repo: Optional[str] = typer.Option(None, "--repo", "-r", help="Default repo slug"),
    defaults: Optional[str] = typer.Option(None, "--defaults", help="JSON object or @file with default fields"),
    concurrency: int = typer.Option(1, "--concurrency", "-c", help="Number of concurrent requests"),
    continue_on_error: bool = typer.Option(True, "--continue-on-error/--stop-on-error"),
    json_out: bool = typer.Option(False, "--json", help="Print raw JSON results"),
):
    """Batch delete blocker comments."""
    _batch_execute(
        file=file,
        project=project,
        repo=repo,
        defaults=defaults,
        concurrency=concurrency,
        continue_on_error=continue_on_error,
        json_out=json_out,
        op=_batch_op_pr_blockers_delete,
    )


@batch_pr_review_app.command("get")
def pr_batch_review_get(
    file: str = typer.Option(..., "--file", "-f", help="Path to JSON list (or '-' for stdin)"),
    project: Optional[str] = typer.Option(None, "--project", "-p", help="Default project key"),
    repo: Optional[str] = typer.Option(None, "--repo", "-r", help="Default repo slug"),
    defaults: Optional[str] = typer.Option(None, "--defaults", help="JSON object or @file with default fields"),
    concurrency: int = typer.Option(1, "--concurrency", "-c", help="Number of concurrent requests"),
    continue_on_error: bool = typer.Option(True, "--continue-on-error/--stop-on-error"),
    json_out: bool = typer.Option(False, "--json", help="Print raw JSON results"),
):
    """Batch get review data for pull requests."""
    _batch_execute(
        file=file,
        project=project,
        repo=repo,
        defaults=defaults,
        concurrency=concurrency,
        continue_on_error=continue_on_error,
        json_out=json_out,
        op=_batch_op_pr_review_get,
    )


@batch_pr_review_app.command("complete")
def pr_batch_review_complete(
    file: str = typer.Option(..., "--file", "-f", help="Path to JSON list (or '-' for stdin)"),
    project: Optional[str] = typer.Option(None, "--project", "-p", help="Default project key"),
    repo: Optional[str] = typer.Option(None, "--repo", "-r", help="Default repo slug"),
    defaults: Optional[str] = typer.Option(None, "--defaults", help="JSON object or @file with default fields"),
    concurrency: int = typer.Option(1, "--concurrency", "-c", help="Number of concurrent requests"),
    continue_on_error: bool = typer.Option(True, "--continue-on-error/--stop-on-error"),
    json_out: bool = typer.Option(False, "--json", help="Print raw JSON results"),
):
    """Batch complete reviews for pull requests."""
    _batch_execute(
        file=file,
        project=project,
        repo=repo,
        defaults=defaults,
        concurrency=concurrency,
        continue_on_error=continue_on_error,
        json_out=json_out,
        op=_batch_op_pr_review_complete,
    )


@batch_pr_review_app.command("discard")
def pr_batch_review_discard(
    file: str = typer.Option(..., "--file", "-f", help="Path to JSON list (or '-' for stdin)"),
    project: Optional[str] = typer.Option(None, "--project", "-p", help="Default project key"),
    repo: Optional[str] = typer.Option(None, "--repo", "-r", help="Default repo slug"),
    defaults: Optional[str] = typer.Option(None, "--defaults", help="JSON object or @file with default fields"),
    concurrency: int = typer.Option(1, "--concurrency", "-c", help="Number of concurrent requests"),
    continue_on_error: bool = typer.Option(True, "--continue-on-error/--stop-on-error"),
    json_out: bool = typer.Option(False, "--json", help="Print raw JSON results"),
):
    """Batch discard reviews for pull requests."""
    _batch_execute(
        file=file,
        project=project,
        repo=repo,
        defaults=defaults,
        concurrency=concurrency,
        continue_on_error=continue_on_error,
        json_out=json_out,
        op=_batch_op_pr_review_discard,
    )


@batch_pr_auto_merge_app.command("get")
def pr_batch_auto_merge_get(
    file: str = typer.Option(..., "--file", "-f", help="Path to JSON list (or '-' for stdin)"),
    project: Optional[str] = typer.Option(None, "--project", "-p", help="Default project key"),
    repo: Optional[str] = typer.Option(None, "--repo", "-r", help="Default repo slug"),
    defaults: Optional[str] = typer.Option(None, "--defaults", help="JSON object or @file with default fields"),
    concurrency: int = typer.Option(1, "--concurrency", "-c", help="Number of concurrent requests"),
    continue_on_error: bool = typer.Option(True, "--continue-on-error/--stop-on-error"),
    json_out: bool = typer.Option(False, "--json", help="Print raw JSON results"),
):
    """Batch get auto-merge requests."""
    _batch_execute(
        file=file,
        project=project,
        repo=repo,
        defaults=defaults,
        concurrency=concurrency,
        continue_on_error=continue_on_error,
        json_out=json_out,
        op=_batch_op_pr_auto_merge_get,
    )


@batch_pr_auto_merge_app.command("set")
def pr_batch_auto_merge_set(
    file: str = typer.Option(..., "--file", "-f", help="Path to JSON list (or '-' for stdin)"),
    project: Optional[str] = typer.Option(None, "--project", "-p", help="Default project key"),
    repo: Optional[str] = typer.Option(None, "--repo", "-r", help="Default repo slug"),
    defaults: Optional[str] = typer.Option(None, "--defaults", help="JSON object or @file with default fields"),
    concurrency: int = typer.Option(1, "--concurrency", "-c", help="Number of concurrent requests"),
    continue_on_error: bool = typer.Option(True, "--continue-on-error/--stop-on-error"),
    json_out: bool = typer.Option(False, "--json", help="Print raw JSON results"),
):
    """Batch request auto-merge for pull requests."""
    _batch_execute(
        file=file,
        project=project,
        repo=repo,
        defaults=defaults,
        concurrency=concurrency,
        continue_on_error=continue_on_error,
        json_out=json_out,
        op=_batch_op_pr_auto_merge_set,
    )


@batch_pr_auto_merge_app.command("cancel")
def pr_batch_auto_merge_cancel(
    file: str = typer.Option(..., "--file", "-f", help="Path to JSON list (or '-' for stdin)"),
    project: Optional[str] = typer.Option(None, "--project", "-p", help="Default project key"),
    repo: Optional[str] = typer.Option(None, "--repo", "-r", help="Default repo slug"),
    defaults: Optional[str] = typer.Option(None, "--defaults", help="JSON object or @file with default fields"),
    concurrency: int = typer.Option(1, "--concurrency", "-c", help="Number of concurrent requests"),
    continue_on_error: bool = typer.Option(True, "--continue-on-error/--stop-on-error"),
    json_out: bool = typer.Option(False, "--json", help="Print raw JSON results"),
):
    """Batch cancel auto-merge for pull requests."""
    _batch_execute(
        file=file,
        project=project,
        repo=repo,
        defaults=defaults,
        concurrency=concurrency,
        continue_on_error=continue_on_error,
        json_out=json_out,
        op=_batch_op_pr_auto_merge_cancel,
    )


@app.command("doctor")
def doctor(
    json_out: bool = typer.Option(False, "--json", help="Print machine-readable JSON status"),
):
    """Sanity checks: validates env vars and hits a lightweight endpoint."""
    bb = client()
    # Hit an endpoint that typically requires only auth and returns quickly.
    # We'll use dashboard pull-requests (even if empty) as a general check.
    resp = bb.request("GET", "dashboard/pull-requests", params={"limit": 1, "start": 0})
    # If we got here, auth + base URL are OK.
    visible = 0
    if isinstance(resp, dict) and "values" in resp and isinstance(resp.get("values"), list):
        visible = len(resp.get("values") or [])

    if json_out:
        _print_json(
            {
                "ok": True,
                "message": "BITBUCKET_SERVER and BITBUCKET_API_TOKEN look usable.",
                "dashboard_prs_visible_first_page": visible,
            }
        )
        return

    typer.echo("OK: BITBUCKET_SERVER and BITBUCKET_API_TOKEN look usable.")
    typer.echo(f"Dashboard PRs visible: {visible} item(s) on first page.")


def main() -> None:
    try:
        app()
    except BBError as e:
        typer.secho(str(e), fg=typer.colors.RED, err=True)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
