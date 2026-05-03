#!/usr/bin/env python3
"""
Fetch GitHub repositories with 1000+ stars and output as structured JSON.

Usage:
    python fetch_popular_repos.py [options]

Options:
    --token TOKEN       GitHub personal access token (or set GITHUB_TOKEN env var)
    --min-stars N       Minimum star count (default: 1000)
    --limit N           Maximum number of repos to fetch (default: 100)
    --language LANG     Filter by programming language (optional)
    --sort FIELD        Sort by: stars, forks, updated (default: stars)
    --output FILE       Output file path (default: stdout)
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


def build_headers(token: str | None) -> dict:
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "fetch-popular-repos/1.0",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def search_repos(
    token: str | None,
    min_stars: int,
    limit: int,
    language: str | None,
    sort: str,
) -> list[dict]:
    headers = build_headers(token)
    results = []
    per_page = min(limit, 100)
    page = 1

    query_parts = [f"stars:>={min_stars}"]
    if language:
        query_parts.append(f"language:{language}")
    query = " ".join(query_parts)

    while len(results) < limit:
        params = urllib.parse.urlencode({
            "q": query,
            "sort": sort,
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
            print(f"GitHub API error {e.code}: {body}", file=sys.stderr)
            sys.exit(1)

        items = data.get("items", [])
        if not items:
            break

        for item in items:
            if len(results) >= limit:
                break
            results.append({
                "rank": len(results) + 1,
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
            })

        page += 1
        # Respect GitHub's secondary rate limit for search (30 req/min)
        if len(results) < limit and len(items) == per_page:
            time.sleep(2)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch GitHub repos with 1000+ stars as JSON"
    )
    parser.add_argument("--token", default=os.environ.get("GITHUB_TOKEN"))
    parser.add_argument("--min-stars", type=int, default=1000)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--language", default=None)
    parser.add_argument("--sort", choices=["stars", "forks", "updated"], default="stars")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    repos = search_repos(
        token=args.token,
        min_stars=args.min_stars,
        limit=args.limit,
        language=args.language,
        sort=args.sort,
    )

    output = {
        "fetched_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "query": {
            "min_stars": args.min_stars,
            "language": args.language,
            "sort": args.sort,
            "limit": args.limit,
        },
        "total": len(repos),
        "repositories": repos,
    }

    json_str = json.dumps(output, ensure_ascii=False, indent=2)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(json_str)
        print(f"Saved {len(repos)} repositories to {args.output}", file=sys.stderr)
    else:
        print(json_str)


if __name__ == "__main__":
    main()
