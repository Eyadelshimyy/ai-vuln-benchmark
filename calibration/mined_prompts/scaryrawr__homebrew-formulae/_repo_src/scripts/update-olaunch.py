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
                check=True,
                capture_output=True,
                text=True,
            )

            sha256_hash = hashlib.sha256()
            with open(temp_file.name, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    sha256_hash.update(chunk)

            return sha256_hash.hexdigest()
        except subprocess.CalledProcessError as e:
            details = e.stderr.strip() if e.stderr else str(e)
            raise Exception(f"Failed to download {url}: {details}")


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


def update_olaunch_formula():
    """Update the olaunch formula with latest release."""
    repo = "scaryrawr/olaunch"
    formula_path = "Formula/olaunch.rb"

    try:
        latest_tag = get_latest_release(repo)
        latest_version = latest_tag.lstrip("v")

        with open(formula_path, "r") as f:
            content = f.read()

        version_match = re.search(r'version "([^"]+)"', content)
        if version_match:
            current_version = version_match.group(1)
        else:
            url_match = re.search(
                r'url "https://github\.com/scaryrawr/olaunch/releases/download/([^/]+)/',
                content,
            )
            if not url_match:
                print("Could not find version or URL in olaunch.rb")
                return False
            current_version = url_match.group(1).lstrip("v")

        if current_version == latest_version:
            print(f"olaunch is already up to date at {latest_version}")
            return False

        targets = [
            "aarch64-apple-darwin",
            "x86_64-unknown-linux-gnu",
            "aarch64-unknown-linux-gnu",
        ]

        updates = []
        for target in targets:
            asset_name = f"olaunch-{latest_tag}-{target}.tar.gz"
            new_url = (
                f"https://github.com/scaryrawr/olaunch/releases/download/{latest_tag}/"
                f"{asset_name}"
            )
            new_sha256 = get_sha256(new_url)
            print(f"Updated {target}: {new_sha256}")
            updates.append((target, new_url, new_sha256))

        new_content = re.sub(
            r'^\s*version "[^"]+"\n',
            "",
            content,
            count=1,
            flags=re.MULTILINE,
        )

        for target, new_url, new_sha256 in updates:
            url_sha_regex = re.compile(
                rf'url "https://github\.com/scaryrawr/olaunch/releases/download/[^/]+/olaunch-[^"]+-{re.escape(target)}\.tar\.gz"\n'
                r'(?:\s*version "[^"]+"\n)?(\s*)sha256 "[^"]+"',
                re.MULTILINE,
            )
            new_content, replacements = url_sha_regex.subn(
                lambda m: f'url "{new_url}"\n{m.group(1)}sha256 "{new_sha256}"',
                new_content,
                count=1,
            )
            if replacements != 1:
                raise Exception(f"Could not update URL and sha256 for {target}")

        with open(formula_path, "w") as f:
            f.write(new_content)

        print(f"Updated olaunch from {current_version} to {latest_version}")
        return True

    except Exception as e:
        print(f"Error updating olaunch: {e}")
        return False


if __name__ == "__main__":
    update_olaunch_formula()
