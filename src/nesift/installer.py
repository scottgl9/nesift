"""Install nesift skill files for supported agent families."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import NamedTuple

_SKILLS_SRC = Path(__file__).parent / "skills"
_TARGET_DIRS = {
    "claude": Path(".claude/skills/nesift"),
    "openclaw": Path(".openclaw/skills/nesift"),
    "agents": Path(".agents/skills/nesift"),
}
_DETECT_ROOTS = {
    "claude": Path(".claude"),
    "openclaw": Path(".openclaw"),
    "agents": Path(".agents"),
}


class InstallResult(NamedTuple):
    target: str
    dest: Path
    files_written: list[Path]
    skipped: bool
    reason: str


def _home_path(path: Path) -> Path:
    return Path.home() / path


def detect_targets() -> list[str]:
    """Return targets whose home-directory roots already exist."""

    return [name for name, root in _DETECT_ROOTS.items() if _home_path(root).exists()]


def _files_for_target(target: str) -> list[tuple[Path, Path]]:
    files = [(_SKILLS_SRC / "SKILL.md", Path("SKILL.md"))]
    if target == "openclaw":
        files.append((_SKILLS_SRC / "openclaw_bridge.py", Path("scripts/nesift.py")))
    return files


def install(targets: list[str], *, force: bool, dry_run: bool) -> list[InstallResult]:
    """Install packaged skill files into the requested targets."""

    results: list[InstallResult] = []
    for target in targets:
        rel_dest = _TARGET_DIRS.get(target)
        if rel_dest is None:
            results.append(
                InstallResult(
                    target=target,
                    dest=Path.home(),
                    files_written=[],
                    skipped=True,
                    reason="unknown target",
                )
            )
            continue

        dest = _home_path(rel_dest)
        file_pairs = _files_for_target(target)
        existing = [dest / rel_path for _, rel_path in file_pairs if (dest / rel_path).exists()]
        if existing and not force:
            names = ", ".join(str(path.relative_to(dest)) for path in existing)
            results.append(
                InstallResult(
                    target=target,
                    dest=dest,
                    files_written=[],
                    skipped=True,
                    reason=f"already exists: {names}",
                )
            )
            continue

        planned = [dest / rel_path for _, rel_path in file_pairs]
        if dry_run:
            results.append(
                InstallResult(
                    target=target,
                    dest=dest,
                    files_written=planned,
                    skipped=False,
                    reason="dry-run",
                )
            )
            continue

        dest.mkdir(parents=True, exist_ok=True)
        written: list[Path] = []
        for src, rel_path in file_pairs:
            out = dest / rel_path
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, out)
            written.append(out)
        results.append(
            InstallResult(
                target=target,
                dest=dest,
                files_written=written,
                skipped=False,
                reason="installed",
            )
        )
    return results


def _claude_desktop_config_path() -> Path:
    if sys.platform == "darwin":
        return Path.home() / "Library/Application Support/Claude/claude_desktop_config.json"
    return Path.home() / ".config/Claude/claude_desktop_config.json"


def install_mcp(*, dry_run: bool) -> tuple[bool, str]:
    """Register nesift-mcp in Claude Desktop's MCP config."""

    exe = shutil.which("nesift-mcp")
    if exe is None:
        return False, "nesift-mcp not found in PATH"

    config_path = _claude_desktop_config_path()
    if config_path.exists():
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            return False, f"invalid Claude Desktop config: {exc}"
    else:
        config = {}

    if not isinstance(config, dict):
        return False, "Claude Desktop config must be a JSON object"

    servers = config.setdefault("mcpServers", {})
    if not isinstance(servers, dict):
        return False, "Claude Desktop config mcpServers must be an object"

    desired = {"command": exe, "args": []}
    existing = servers.get("nesift-mcp")
    if existing == desired:
        return True, f"nesift-mcp already configured in {config_path}"

    servers["nesift-mcp"] = desired
    if dry_run:
        return True, f"would update {config_path}"

    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return True, f"updated {config_path}"
