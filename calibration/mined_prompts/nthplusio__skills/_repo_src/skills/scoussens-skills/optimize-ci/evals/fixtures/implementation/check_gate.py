from gate import aggregate


def check(condition, message):
    if not condition:
        raise AssertionError(message)


expected = ["api", "web"]
check(aggregate(expected, {"api": "success", "web": "success"}), "success")
check(aggregate(expected, {"api": "success", "web": "skipped"}, True, ("web",)), "valid skip")
check(not aggregate(expected, {"api": "success", "web": "skipped"}, False), "invalid skip")
check(not aggregate(expected, {"api": "skipped", "web": "success"}, True, ("web",)), "skip not authorized for task")
check(not aggregate(expected, {"api": "success", "web": "failed"}), "failed prerequisite")
check(not aggregate(expected, {"api": "success", "web": "cancelled"}), "cancelled prerequisite")
check(not aggregate(expected, {"api": "success"}), "missing prerequisite")
check(not aggregate(expected, {"unrelated": "success"}), "unrelated success")
check(not aggregate(expected, {"api": "success", "web": "neutral"}), "unknown verdict")
print("aggregate gate checks passed")
