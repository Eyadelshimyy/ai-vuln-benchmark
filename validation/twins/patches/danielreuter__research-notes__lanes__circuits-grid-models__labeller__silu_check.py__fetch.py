def fetch(item: str) -> Path:
    if not item or Path(item).name != item or item in (".", ".."):
        raise ValueError("item must be a single path component")
    here = sorted(Path(f"/workspace/jobs/cov/{item}").glob("*/commit"))
    if here and (here[0] / "replay_slim_p0" / "slim.json").exists():
        return here[0]
    d = Path("/tmp/gm/dive") / item
    if not (d / "replay_slim_p0" / "slim.json").exists():
        d.mkdir(parents=True, exist_ok=True)
        ssh = subprocess.run([RESEARCH, "pods", "ssh", "vy-nebius-1", "--print"], capture_output=True, text=True, check=True).stdout.split()
        tar = subprocess.run([*ssh, f"cd /workspace/jobs/cov/{item}/*/commit && tar -cf - replay_slim_p0 sampled_replay_p0.json"],
                             capture_output=True, check=True, timeout=300).stdout
        subprocess.run(["tar", "-C", str(d), "-xf", "-"], input=tar, check=True)
    return d
