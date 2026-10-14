#!/usr/bin/env python3

import subprocess
import hashlib
import re
import tempfile


def get_sha256(url):
    """Download file and calculate SHA256 checksum using curl."""
    with tempfile.NamedTemporaryFile() as temp_file:
        try:
            subprocess.run(["curl", "-sL", url, "-o", temp_file.name], check=True)

            sha256_hash = hashlib.sha256()
            with open(temp_file.name, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    sha256_hash.update(chunk)

            return sha256_hash.hexdigest()
        except subprocess.CalledProcessError as e:
            raise Exception(f"Failed to download {url}: {e}")


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
        raise Exception(f"Failed to get latest release for {repo}: {e}")


def update_sl_formula():
    """Update the sl formula with latest release."""
    repo = "scaryrawr/sl"
    formula_path = "Formula/sl.rb"

    try:
        # Get latest release
        latest_tag = get_latest_release(repo)

        # Read current formula
        with open(formula_path, "r") as f:
            content = f.read()

        # Extract current version from URL
        url_match = re.search(
            r'url "https://github\.com/scaryrawr/sl/archive/refs/tags/([^"]+)\.tar\.gz"',
            content,
        )
        if not url_match:
            print("Could not find URL in sl.rb")
            return False

        current_tag = url_match.group(1)

        if current_tag == latest_tag:
            print(f"sl is already up to date at {latest_tag}")
            return False

        # Calculate new URL and SHA256
        new_url = (
            f"https://github.com/scaryrawr/sl/archive/refs/tags/{latest_tag}.tar.gz"
        )
        new_sha256 = get_sha256(new_url)

        # Update formula
        new_content = re.sub(
            r'url "https://github\.com/scaryrawr/sl/archive/refs/tags/[^"]+\.tar\.gz"',
            f'url "{new_url}"',
            content,
        )
        new_content = re.sub(r'sha256 "[^"]+"', f'sha256 "{new_sha256}"', new_content)

        # Write updated formula
        with open(formula_path, "w") as f:
            f.write(new_content)

        print(f"Updated sl from {current_tag} to {latest_tag}")
        return True

    except Exception as e:
        print(f"Error updating sl: {e}")
        return False


if __name__ == "__main__":
    update_sl_formula()
