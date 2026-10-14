#!/usr/bin/env python3

import hashlib
import re
import subprocess
import tempfile


def get_sha256(url):
    """Download file and calculate SHA256 checksum using curl."""
    with tempfile.NamedTemporaryFile() as temp_file:
        try:
            subprocess.run(
                ["curl", "-fsSL", url, "-o", temp_file.name],
                capture_output=True,
                text=True,
                check=True,
            )

            sha256_hash = hashlib.sha256()
            with open(temp_file.name, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    sha256_hash.update(chunk)

            return sha256_hash.hexdigest()
        except subprocess.CalledProcessError as e:
            detail = e.stderr.strip() or str(e)
            raise Exception(f"Failed to download {url}: {detail}")


def get_latest_release(repo):
    """Get the latest release tag from GitHub using gh CLI."""
    try:
        result = subprocess.run(
            ["gh", "api", f"repos/{repo}/releases/latest", "--jq", ".tag_name"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        detail = e.stderr.strip() or e.stdout.strip() or str(e)
        if "404" in detail or "Not Found" in detail:
            return None
        raise Exception(f"Failed to get latest release for {repo}: {detail}")


def get_latest_tag(repo):
    """Get the latest tag from GitHub using gh CLI."""
    try:
        result = subprocess.run(
            ["gh", "api", f"repos/{repo}/tags", "--jq", ".[0].name"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        detail = e.stderr.strip() or e.stdout.strip() or str(e)
        raise Exception(f"Failed to get latest tag for {repo}: {detail}")


def get_latest_version_tag(repo):
    """Prefer the latest release tag, falling back to tags for repos without releases."""
    return get_latest_release(repo) or get_latest_tag(repo)


def update_smahtutils_formula():
    """Update the smahtutils formula with latest tag."""
    repo = "scaryrawr/smahtutils"
    formula_path = "Formula/smahtutils.rb"

    try:
        latest_tag = get_latest_version_tag(repo)

        with open(formula_path, "r") as f:
            content = f.read()

        url_match = re.search(
            r'url "https://github\.com/scaryrawr/smahtutils/archive/refs/tags/([^"]+)\.tar\.gz"',
            content,
        )
        if not url_match:
            print("Could not find URL in smahtutils.rb")
            return False

        current_tag = url_match.group(1)

        if current_tag == latest_tag:
            print(f"smahtutils is already up to date at {latest_tag}")
            return False

        new_url = f"https://github.com/scaryrawr/smahtutils/archive/refs/tags/{latest_tag}.tar.gz"
        new_sha256 = get_sha256(new_url)

        new_content = re.sub(
            r'url "https://github\.com/scaryrawr/smahtutils/archive/refs/tags/[^"]+\.tar\.gz"',
            f'url "{new_url}"',
            content,
        )
        new_content = re.sub(r'sha256 "[^"]+"', f'sha256 "{new_sha256}"', new_content)

        with open(formula_path, "w") as f:
            f.write(new_content)

        print(f"Updated smahtutils from {current_tag} to {latest_tag}")
        return True

    except Exception as e:
        print(f"Error updating smahtutils: {e}")
        return False


if __name__ == "__main__":
    update_smahtutils_formula()
