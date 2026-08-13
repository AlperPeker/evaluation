import subprocess
import tomlkit

from shared import log

STARGZ_CONFIG_PATH = "/etc/containerd-stargz-grpc/config.toml"


def read_base_config() -> str:
    result = subprocess.run(
        ["sudo", "cat", STARGZ_CONFIG_PATH],
        capture_output=True, text=True, check=True,
    )
    return result.stdout


def apply_overrides(base_content: str, overrides: dict) -> str:
    """Set each override in the right TOML table, preserving comments/formatting.

    Keys are dotted paths, e.g. "fuse.passthrough" targets [fuse]; a bare key
    targets the root table. Missing tables are created.
    """
    doc = tomlkit.parse(base_content)
    for dotted_key, value in overrides.items():
        *tables, leaf = dotted_key.split(".")
        node = doc
        for table in tables:
            if table not in node:
                node[table] = tomlkit.table()
            node = node[table]
        node[leaf] = value
    return tomlkit.dumps(doc)


def apply_stargz_config(config_content: str) -> None:
    """Stop service, write config, start service."""
    current = read_base_config()
    log.info("--- applying stargz config ---")
    log.info(f"BEFORE:\n{current}")
    log.info(f"AFTER:\n{config_content}")
    tmp = "/tmp/stargz-config-measure.toml"
    with open(tmp, "w") as f:
        f.write(config_content)
    subprocess.run(["sudo", "systemctl", "stop", "stargz-snapshotter"], check=True)
    subprocess.run(["sudo", "cp", tmp, STARGZ_CONFIG_PATH], check=True)
    subprocess.run(["sudo", "systemctl", "start", "stargz-snapshotter"], check=True)
