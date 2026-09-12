"""
Cross-repo test fixture generator.

Creates three tiny local git repos under FIXTURES_DIR, deliberately wired
to exercise all three CrossRepoLinker strategies in one test run:

  orders-service        -> has an OpenAPI spec for GET /orders/{id}.
                            No outbound calls. Target for the CONFIRMED case.

  checkout-service       -> calls orders-service via a URL built from an
                            f-string + a variable (same pattern as your
                            real call_n8n_webhook: base URL in a variable,
                            path segment interpolated). This should match
                            orders-service's spec -> CONFIRMED edge.
                            Also calls notifications-service, which has NO
                            spec -> should fall through to the repo-name
                            match -> INFERRED edge.

  notifications-service -> no spec, no outbound calls. Just a link target.

A cross_repo_links.yaml is also written declaring a MANUAL override link
between orders-service and notifications-service (which have no real code
connection to each other) — this is purely to verify the override
mechanism fires independently of the other two strategies.

Usage:
    python setup_cross_repo_fixtures.py [fixtures_dir]

Each repo is git-initialized on branch 'main' and committed, so
RepoRegistry.register() can clone them via a plain local filesystem path
(git supports `git clone <local_path>` same as a remote URL).
"""
import os
import sys
import subprocess

FIXTURES_DIR = sys.argv[1] if len(sys.argv) > 1 else "./cross_repo_fixtures"

REPOS = {
    "orders-service": {
        "openapi.yaml": """\
openapi: 3.0.0
info:
  title: orders-service
  version: "1.0"
paths:
  /orders/{id}:
    get:
      operationId: getOrder
      responses:
        '200':
          description: OK
""",
        "app.py": """\
def get_order(order_id: str) -> dict:
    # Stub — the OpenAPI spec above is what the Cross-Repo Linker actually
    # reads; this function just makes the repo look like a real service.
    return {"id": order_id, "status": "confirmed"}
""",
    },
    "checkout-service": {
        "client.py": """\
import requests

ORDERS_BASE_URL = "http://orders-service:8000"


def fetch_order(order_id: str):
    # Same shape as call_n8n_webhook: base URL in a variable, path
    # segment interpolated via f-string. Exercises identifier resolution
    # + f-string wildcarding together, same as the real repo did.
    url = f"{ORDERS_BASE_URL}/orders/{order_id}"
    return requests.get(url)


def notify_user(user_id: str):
    # notifications-service has no OpenAPI spec, so this should fall
    # through to the repo-name-match fallback (INFERRED), not a confirmed
    # spec match.
    url = f"http://notifications-service:8000/notify/{user_id}"
    return requests.post(url, json={})
""",
    },
    "notifications-service": {
        "app.py": """\
def send_notification(user_id: str, message: str) -> None:
    # Stub — no spec, no outbound calls. Exists purely as a link target
    # for the inferred-match and manual-override test cases.
    print(f"notifying {user_id}: {message}")
""",
    },
}

MANUAL_OVERRIDE_CONFIG = """\
links:
  - source_repo: orders-service
    target_repo: notifications-service
    edge_type: manual
"""


def run(cmd, cwd):
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{result.stderr}")


def main():
    os.makedirs(FIXTURES_DIR, exist_ok=True)

    for repo_name, files in REPOS.items():
        repo_dir = os.path.join(FIXTURES_DIR, repo_name)
        os.makedirs(repo_dir, exist_ok=True)
        for filename, content in files.items():
            with open(os.path.join(repo_dir, filename), "w") as f:
                f.write(content)

        run(["git", "init", "-b", "main"], cwd=repo_dir)
        run(["git", "config", "user.email", "fixture@example.com"], cwd=repo_dir)
        run(["git", "config", "user.name", "fixture-generator"], cwd=repo_dir)
        run(["git", "add", "."], cwd=repo_dir)
        run(["git", "commit", "-m", "initial fixture commit"], cwd=repo_dir)
        print(f"Created and committed: {os.path.abspath(repo_dir)}")

    override_path = os.path.join(os.getcwd(), "cross_repo_links.yaml")
    with open(override_path, "w") as f:
        f.write(MANUAL_OVERRIDE_CONFIG)
    print(f"\nWrote manual override config: {override_path}")
    print("(CROSS_REPO_LINKS_CONFIG env var must point here, or leave unset "
          "since this matches the default './cross_repo_links.yaml')")

    print("\nDone. Register each with RepoRegistry using its local path as git_url, e.g.:")
    for repo_name in REPOS:
        print(f'  registry.register(name="{repo_name}", '
              f'git_url=r"{os.path.abspath(os.path.join(FIXTURES_DIR, repo_name))}", '
              f'default_branch="main")')


if __name__ == "__main__":
    main()