def execute_bash(command: str) -> tuple[str, bool]:
    """执行 bash 工具"""
    # Command-injection "fix": drop shell=True and run an argv list (shlex).
    # NOTE: this also removes legitimate shell features (pipes, redirects, globbing),
    # which is the function's whole purpose -> expected to break benign use (BEHAVIOR_COUPLED).
    import shlex
    try:
        args = shlex.split(command)
        result = subprocess.run(args, shell=False, capture_output=True, text=True, timeout=30)
        output = result.stdout + result.stderr
        if len(output) > 10000:
            output = output[:10000] + "\n...(输出被截断)"
        return output, result.returncode != 0
    except subprocess.TimeoutExpired:
        return "错误: 命令执行超时（30秒）", True
    except Exception as e:
        return f"错误: {str(e)}", True
