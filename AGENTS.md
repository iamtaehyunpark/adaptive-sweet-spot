<!-- homemade:start -->
## Homemade

This folder is registered in **Homemade** (the personal-software launcher) as app `ass-web` — mode `webapp`, exec `python3 tools/app.py --port 8765`.
The manifest is `homemade.toml` (keep it in sync with any CLI argument you add or change; declared inputs: camera (string), dsp (string), no_camera (string)).
Follow the `homemade` skill (`/Users/t/.homemade/skills/homemade/SKILL.md`) for conventions: uv, `$HOMEMADE_OUTPUT_DIR`, `@homemade:progress` / `@homemade:artifact`.

You can drive the launcher yourself instead of asking the user to click Run:
- CLI: `/Users/t/.homemade/bin/homemade` (`homemade run ass-web key=value --wait`, `homemade activity ass-web`, `homemade log <run-id> --follow`,
  `homemade top|ps|signal|stdin|capture|record`, `homemade help`).
- MCP: `/Applications/Homemade.app/Contents/MacOS/homemade --mcp` (config file `/Users/t/.homemade/mcp.json`; tools: start_run, wait_run, app_activity, get_run_log, last_failed_run, explain_run,
  list_artifacts, apply_manifest, git_checkpoint, open_app, capture, …). Register once: `claude mcp add homemade -- /Applications/Homemade.app/Contents/MacOS/homemade --mcp`.
- Inside Homemade sessions these are already on PATH / connected, and the user can forward `[homemade]` status lines (run started / finished / diagnosis) into the session.
<!-- homemade:end -->
