#!/usr/bin/env python3
"""
Fetch GitHub repositories with 1000+ stars and output as structured JSON.

GitHub Search API returns at most 1000 results per query. To fetch beyond that
limit, this script uses star-count range windows as a cursor: each window query
fetches up to 1000 repos, then the next window starts below the minimum star
count of the previous batch.

Results are written in batches of --batch-size (default 1000) to separate files:
  <output-dir>/repos_000001-001000.json
  <output-dir>/repos_001001-002000.json
  ...

Use --offset to resume from a global rank position (skip already-saved batches).

Usage:
    python fetch_popular_repos.py [options]

Options:
    --token TOKEN        GitHub personal access token (or GITHUB_TOKEN env var)
    --min-stars N        Minimum star count (default: 1000)
    --max-stars N        Maximum star count for first window (default: 999999999)
    --total-limit N      Stop after this many total repos (default: 10000)
    --batch-size N       Repos per output file (default: 1000)
    --offset N           Skip the first N global ranks; use to resume (default: 0)
    --language LANG      Filter by programming language (optional)
    --output-dir DIR     Directory for output files (default: ./output)
    --delay SECS         Seconds to wait between API requests (default: 2)
"""

import argparse
import json
import os
import sys
import time
import urllib.request
import urllib.parse
import urllib.error
from datetime import datetime


GITHUB_SEARCH_MAX = 1000  # GitHub hard limit per query


def build_headers(token: str | None) -> dict:
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "fetch-popular-repos/1.0",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def fetch_window(
    headers: dict,
    min_stars: int,
    max_stars: int,
    language: str | None,
    delay: float,
) -> list[dict]:
    """Fetch up to 1000 repos in the star range [min_stars, max_stars], stars desc."""
    results = []
    page = 1

    query_parts = [f"stars:{min_stars}..{max_stars}"]
    if language:
        query_parts.append(f"language:{language}")
    query = " ".join(query_parts)

    while len(results) < GITHUB_SEARCH_MAX:
        remaining = GITHUB_SEARCH_MAX - len(results)
        per_page = min(remaining, 100)

        params = urllib.parse.urlencode({
            "q": query,
            "sort": "stars",
            "order": "desc",
            "per_page": per_page,
            "page": page,
        })
        url = f"https://api.github.com/search/repositories?{params}"

        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req) as resp:
                data = json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            body = e.read().decode()
            error = json.loads(body) if body.startswith("{") else {"message": body}
            if e.code == 403 and "rate limit" in error.get("message", "").lower():
                print("Rate limit hit, sleeping 60s...", file=sys.stderr)
                time.sleep(60)
                continue
            print(f"GitHub API error {e.code}: {body}", file=sys.stderr)
            sys.exit(1)

        items = data.get("items", [])
        if not items:
            break

        results.extend(items)
        page += 1

        if len(items) < per_page:
            break  # last page

        time.sleep(delay)

    return results


def to_repo_record(item: dict, global_rank: int) -> dict:
    return {
        "rank": global_rank,
        "name": item["full_name"],
        "url": item["html_url"],
        "description": item.get("description") or "",
        "stars": item["stargazers_count"],
        "forks": item["forks_count"],
        "language": item.get("language") or "",
        "topics": item.get("topics", []),
        "created_at": item["created_at"],
        "updated_at": item["updated_at"],
        "open_issues": item["open_issues_count"],
        "license": (item.get("license") or {}).get("spdx_id") or "",
    }


def save_batch(
    batch: list[dict],
    output_dir: str,
    global_start: int,
    fetched_at: str,
    args_snapshot: dict,
) -> str:
    global_end = global_start + len(batch) - 1
    filename = f"repos_{global_start:06d}-{global_end:06d}.json"
    path = os.path.join(output_dir, filename)

    payload = {
        "fetched_at": fetched_at,
        "query": args_snapshot,
        "range": {"from": global_start, "to": global_end},
        "total": len(batch),
        "repositories": batch,
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch GitHub repos with 1000+ stars, batched JSON output"
    )
    parser.add_argument("--token", default=os.environ.get("GITHUB_TOKEN"))
    parser.add_argument("--min-stars", type=int, default=1000)
    parser.add_argument("--max-stars", type=int, default=999_999_999)
    parser.add_argument("--total-limit", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--offset", type=int, default=0,
                        help="Resume from this global rank (1-based); skips earlier results")
    parser.add_argument("--language", default=None)
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--delay", type=float, default=2.0)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    headers = build_headers(args.token)
    fetched_at = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

    args_snapshot = {
        "min_stars": args.min_stars,
        "max_stars": args.max_stars,
        "language": args.language,
        "total_limit": args.total_limit,
        "batch_size": args.batch_size,
        "offset": args.offset,
    }

    current_max = args.max_stars
    global_rank = 1          # tracks rank across all windows (before offset skip)
    saved_count = 0          # repos actually written to files
    pending_batch: list[dict] = []
    batch_start_rank = args.offset + 1  # global rank of first item in current batch

    print(
        f"Fetching repos with {args.min_stars}+ stars, "
        f"offset={args.offset}, limit={args.total_limit}",
        file=sys.stderr,
    )

    while saved_count < args.total_limit:
        if current_max < args.min_stars:
            break

        print(
            f"  Window: stars<={current_max}  (collected so far: "
            f"{args.offset + saved_count})",
            file=sys.stderr,
        )

        window_items = fetch_window(
            headers=headers,
            min_stars=args.min_stars,
            max_stars=current_max,
            language=args.language,
            delay=args.delay,
        )

        if not window_items:
            break

        for item in window_items:
            # Skip items before the requested offset
            if global_rank <= args.offset:
                global_rank += 1
                continue

            record = to_repo_record(item, global_rank)
            pending_batch.append(record)
            global_rank += 1
            saved_count += 1

            # Flush a full batch to disk
            if len(pending_batch) >= args.batch_size:
                path = save_batch(
                    pending_batch, args.output_dir,
                    batch_start_rank, fetched_at, args_snapshot,
                )
                print(f"  Saved: {path}", file=sys.stderr)
                batch_start_rank += len(pending_batch)
                pending_batch = []

            if saved_count >= args.total_limit:
                break

        # Advance window cursor: next window starts below the min stars of this one
        min_stars_in_window = min(item["stargazers_count"] for item in window_items)
        next_max = min_stars_in_window - 1

        if next_max >= current_max:
            # Safeguard: all items had the same star count; step down by 1
            next_max = current_max - 1

        current_max = next_max
        time.sleep(args.delay)

    # Flush remaining partial batch
    if pending_batch:
        path = save_batch(
            pending_batch, args.output_dir,
            batch_start_rank, fetched_at, args_snapshot,
        )
        print(f"  Saved: {path}", file=sys.stderr)

    print(
        f"Done. Total saved: {saved_count} repos across "
        f"{args.output_dir}/",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
