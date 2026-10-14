#!/usr/bin/env python3

import hashlib
import re
import subprocess
import tempfile


def get_sha256(url):
    """Download a file and calculate its SHA256 checksum."""
    with tempfile.NamedTemporaryFile() as temp_file:
        try:
            subprocess.run(["curl", "-fsSL", url, "-o", temp_file.name], check=True)
        except subprocess.CalledProcessError as e:
            raise Exception(f"Failed to download {url}: {e}")

        sha256_hash = hashlib.sha256()
        with open(temp_file.name, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()


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


def update_devpod_formula():
    """Update the devpod formula with the latest release."""
    repo = "scaryrawr/devpod"
    formula_path = "Formula/devpod.rb"

    try:
        latest_tag = get_latest_release(repo)
        latest_version = latest_tag.lstrip("v")

        with open(formula_path, "r") as f:
            content = f.read()

        version_match = re.search(r'version "([^"]+)"', content)
        if not version_match:
            print("Could not find version in devpod.rb")
            return False

        current_version = version_match.group(1)
        if current_version == latest_version:
            print(f"devpod is already up to date at {latest_version}")
            return False

        new_content = re.sub(
            r'version "[^"]+"', f'version "{latest_version}"', content
        )

        for platform, arch in [
            ("darwin", "arm64"),
            ("darwin", "amd64"),
            ("linux", "arm64"),
            ("linux", "amd64"),
        ]:
            asset_name = f"devpod-{platform}-{arch}"
            new_url = (
                f"https://github.com/{repo}/releases/download/"
                f"{latest_tag}/{asset_name}"
            )
            new_sha256 = get_sha256(new_url)
            print(f"Updated {platform}-{arch}: {new_sha256}")

            url_sha_regex = re.compile(
                rf'url "https://github\.com/{re.escape(repo)}/releases/download/'
                rf'[^/]+/{asset_name}"\n(\s*)sha256 "[^"]+"',
                re.MULTILINE,
            )
            new_content, replacements = url_sha_regex.subn(
                lambda match: (
                    f'url "{new_url}"\n{match.group(1)}sha256 "{new_sha256}"'
                ),
                new_content,
                count=1,
            )
            if replacements != 1:
                raise Exception(f"Could not find {asset_name} in devpod.rb")

        with open(formula_path, "w") as f:
            f.write(new_content)

        print(f"Updated devpod from {current_version} to {latest_version}")
        return True
    except Exception as e:
        print(f"Error updating devpod: {e}")
        return False


if __name__ == "__main__":
    update_devpod_formula()
