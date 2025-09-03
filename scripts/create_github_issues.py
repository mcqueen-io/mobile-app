import os
import json
import sys
import requests
from typing import List, Dict

"""
Usage:
  - Set GITHUB_TOKEN in environment (with repo scope)
  - Run: python scripts/create_github_issues.py <owner> <repo> .github/speaker_id_issues.json
"""

API_URL = "https://api.github.com"


def create_issue(owner: str, repo: str, token: str, title: str, body: str, labels: List[str]) -> Dict:
    url = f"{API_URL}/repos/{owner}/{repo}/issues"
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    payload = {
        "title": title,
        "body": body,
        "labels": labels or [],
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=30)
    if resp.status_code >= 300:
        raise RuntimeError(f"GitHub API error {resp.status_code}: {resp.text}")
    return resp.json()


def main():
    if len(sys.argv) != 4:
        print("Usage: python scripts/create_github_issues.py <owner> <repo> <json_file>")
        sys.exit(1)

    owner = sys.argv[1]
    repo = sys.argv[2]
    json_path = sys.argv[3]

    token = os.getenv("GITHUB_TOKEN")
    if not token:
        print("Error: GITHUB_TOKEN environment variable not set.")
        sys.exit(2)

    if not os.path.exists(json_path):
        print(f"Error: JSON file not found: {json_path}")
        sys.exit(3)

    with open(json_path, "r", encoding="utf-8") as f:
        issues = json.load(f)

    if not isinstance(issues, list):
        print("Error: JSON must be a list of issue objects")
        sys.exit(4)

    created = []
    for issue in issues:
        title = issue.get("title")
        body = issue.get("body", "")
        labels = issue.get("labels", [])
        if not title:
            print("Skipping issue without title")
            continue
        data = create_issue(owner, repo, token, title, body, labels)
        created.append({"number": data.get("number"), "title": data.get("title")})
        print(f"Created issue #{data.get('number')}: {data.get('title')}")

    print(f"Done. Created {len(created)} issues.")


if __name__ == "__main__":
    main()
