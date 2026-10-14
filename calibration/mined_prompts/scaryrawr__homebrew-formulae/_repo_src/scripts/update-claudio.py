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


def update_claudio_formula():
    """Update the claudio formula with latest release."""
    repo = "scaryrawr/claudio"
    formula_path = "Formula/claudio.rb"

    try:
        latest_tag = get_latest_release(repo)

        with open(formula_path, "r") as f:
            content = f.read()

        version_match = re.search(r'version "([^"]+)"', content)
        if not version_match:
            print("Could not find version in claudio.rb")
            return False

        current_version = version_match.group(1)
        latest_version = latest_tag.lstrip("v")

        if current_version == latest_version:
            print(f"claudio is already up to date at {latest_version}")
            return False

        targets = [
            "aarch64-apple-darwin",
            "x86_64-unknown-linux-gnu",
            "aarch64-unknown-linux-gnu",
        ]

        new_content = re.sub(r'version "[^"]+"', f'version "{latest_version}"', content)

        for target in targets:
            asset_name = f"claudio-{latest_tag}-{target}.tar.gz"
            new_url = (
                f"https://github.com/scaryrawr/claudio/releases/download/{latest_tag}/{asset_name}"
            )

            try:
                new_sha256 = get_sha256(new_url)
                print(f"Updated {target}: {new_sha256}")

                url_sha_regex = re.compile(
                    rf'url "https://github\.com/scaryrawr/claudio/releases/download/[^/]+/claudio-[^"]+-{re.escape(target)}\.tar\.gz"\n(\s*)sha256 "[^"]+"',
                    re.MULTILINE,
                )
                new_content = url_sha_regex.sub(
                    lambda m: f'url "{new_url}"\n{m.group(1)}sha256 "{new_sha256}"',
                    new_content,
                    count=1,
                )

            except Exception as e:
                print(f"Warning: Could not download {target} binary: {e}")
                continue

        with open(formula_path, "w") as f:
            f.write(new_content)

        print(f"Updated claudio from {current_version} to {latest_version}")
        return True

    except Exception as e:
        print(f"Error updating claudio: {e}")
        return False


if __name__ == "__main__":
    update_claudio_formula()
