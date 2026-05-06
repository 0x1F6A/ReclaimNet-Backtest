#!/usr/bin/env python3

from __future__ import annotations

import argparse
import datetime as dt
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import sys
import tempfile
import uuid
import zipfile


NS = 1_000_000_000
DAY_NS = 86_400 * NS
EPOCH = dt.datetime(1970, 1, 1)
INPUT_FILES = (
    "manifest.json",
    "nodes.json",
    "topology.json",
    "jobs.jsonl",
    "reclaims.jsonl",
    "job_segments.jsonl",
    "telemetry.jsonl",
    "bandwidth_samples.jsonl",
    "core_bandwidth_samples.jsonl",
    "hazard_samples.jsonl",
    "historical_migration_profiles.jsonl",
)
OPTIONAL_FILES = ("schema.json",)
ISO_RE = re.compile(
    r"(\d{4})-(\d{2})-(\d{2})[T ](\d{2}):(\d{2}):(\d{2})" r"(?:\.(\d{1,9}))?(Z|[+-]\d{2}:\d{2})?\Z"
)
SAFE_NODE_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")


class PreparationError(ValueError):
    pass


def parse_iso(text: str) -> tuple[int, int]:
    match = ISO_RE.fullmatch(text)
    if not match:
        raise PreparationError(f"invalid ISO-8601 timestamp: {text!r}")
    year, month, day, hour, minute, second = map(int, match.groups()[:6])
    try:
        civil = dt.datetime(year, month, day, hour, minute, second)
    except ValueError as error:
        raise PreparationError(f"invalid ISO-8601 timestamp {text!r}: {error}") from error
    offset = match.group(8)
    offset_s = 0
    if offset and offset != "Z":
        hours, minutes = map(int, offset[1:].split(":"))
        if hours > 23 or minutes > 59:
            raise PreparationError(f"invalid UTC offset in {text!r}")
        offset_s = (hours * 3600 + minutes * 60) * (-1 if offset[0] == "-" else 1)
    fraction = int((match.group(7) or "").ljust(9, "0"))
    delta = civil - EPOCH
    timestamp = (delta.days * 86_400 + delta.seconds - offset_s) * NS + fraction
    if not -(2**63) <= timestamp < 2**63:
        raise PreparationError(
            f"timestamp outside replay's signed 64-bit nanosecond range: {text!r}"
        )
    return timestamp, offset_s


def format_iso(timestamp: int, offset_s: int) -> str:
    seconds, fraction = divmod(timestamp + offset_s * NS, NS)
    civil = EPOCH + dt.timedelta(seconds=seconds)
    sign = "-" if offset_s < 0 else "+"
    absolute = abs(offset_s)
    return (
        f"{civil:%Y-%m-%dT%H:%M:%S}.{fraction:09d}"
        f"{sign}{absolute // 3600:02d}:{absolute // 60 % 60:02d}"
    )


def parse_bound(text: str, origin_ns: int, offset_s: int) -> int:
    if text.startswith("day:"):
        if not re.fullmatch(r"day:[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", text):
            raise PreparationError(f"invalid day index: {text!r}")

        local_day = (origin_ns + offset_s * NS) // DAY_NS
        try:
            day_value = Decimal(text[4:])
            if not day_value.is_finite() or day_value.copy_abs() > Decimal(2**64) / DAY_NS:
                raise PreparationError(f"day index outside replay time range: {text!r}")
            day_offset = day_value * DAY_NS
            result = (
                local_day * DAY_NS
                - offset_s * NS
                + int(day_offset.to_integral_value(rounding=ROUND_HALF_UP))
            )
        except InvalidOperation as error:
            raise PreparationError(f"invalid day index: {text!r}") from error
        if not -(2**63) <= result < 2**63:
            raise PreparationError(f"day index outside replay time range: {text!r}")
        return result
    return parse_iso(text)[0]


def safe_members(archive: zipfile.ZipFile) -> dict[str, zipfile.ZipInfo]:
    members = {}
    for info in archive.infolist():
        name = info.filename
        path = PurePosixPath(name)
        if (
            not name
            or "\\" in name
            or "\x00" in name
            or path.is_absolute()
            or any(part in ("", ".", "..") for part in name.rstrip("/").split("/"))
            or ":" in path.parts[0]
        ):
            raise PreparationError(f"unsafe ZIP member path: {name!r}")
        if stat.S_ISLNK(info.external_attr >> 16):
            raise PreparationError(f"ZIP symbolic links are unsupported: {name!r}")
        normalized = str(path)
        if normalized in members:
            raise PreparationError(f"duplicate ZIP member: {name!r}")
        members[normalized] = info
    return members


def discover_traces(members: dict[str, zipfile.ZipInfo]) -> list[str]:
    return sorted(
        name.removesuffix("/manifest.json")
        for name in members
        if len(PurePosixPath(name).parts) == 2 and name.endswith("/manifest.json")
    )


def read_json(path: Path):
    with path.open(encoding="utf-8") as source:
        return json.load(source)


def write_json(path: Path, value) -> None:
    with path.open("w", encoding="utf-8") as target:
        json.dump(value, target, ensure_ascii=False, allow_nan=False, indent=2)
        target.write("\n")


def records(path: Path):
    with path.open(encoding="utf-8") as source:
        for number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except (ValueError, TypeError) as error:
                raise PreparationError(f"{path.name}:{number}: invalid JSON: {error}") from error
            if not isinstance(value, dict):
                raise PreparationError(f"{path.name}:{number}: expected a JSON object")
            yield value


def file_digest(path: Path) -> dict:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
            size += len(block)
    return {"bytes": size, "sha256": digest.hexdigest()}


def checked_number(value, field: str, minimum=0, maximum=None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PreparationError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < minimum or (maximum is not None and result > maximum):
        raise PreparationError(f"{field} is outside its valid range")
    return result


def checked_timestamp(value, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not -(2**63) <= value < 2**63:
        raise PreparationError(f"{field} must be signed 64-bit epoch nanoseconds")
    return value


def validate_inventory(nodes) -> set[str]:
    if not isinstance(nodes, list) or not nodes:
        raise PreparationError("nodes.json must contain a non-empty node array")
    expected = set()
    for node in nodes:
        node_id = node.get("node_id") if isinstance(node, dict) else None
        if not isinstance(node_id, str) or not SAFE_NODE_RE.fullmatch(node_id):
            raise PreparationError(f"unsafe or missing node_id: {node_id!r}")
        if node_id in expected:
            raise PreparationError(f"duplicate trace node: {node_id}")
        expected.add(node_id)
        count = node.get("gpu_count")
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise PreparationError(f"{node_id}: gpu_count must be positive")
        checked_number(node.get("total_vram_bytes"), f"{node_id}.total_vram_bytes", minimum=1)
    return expected


def validate_mapping(nodes, testbed: dict) -> dict[str, dict]:
    expected = validate_inventory(nodes)
    if not isinstance(testbed, dict) or not isinstance(testbed.get("nodes"), list):
        raise PreparationError("testbed must contain a nodes array")
    if testbed.get("_template"):
        raise PreparationError(
            "testbed is an unfilled template; fill physical hosts and interfaces, "
            "then set _template=false"
        )
    coordinator = testbed.get("coordinator", {})
    for field in ("host", "ssh", "address"):
        if "REPLACE_" in str(coordinator.get(field, "")):
            raise PreparationError(
                f"testbed coordinator.{field} still contains a template placeholder"
            )
    mapping = {}
    for node in testbed["nodes"]:
        if not isinstance(node, dict):
            raise PreparationError("testbed nodes must be objects")
        name = node.get("trace_node")
        if not isinstance(name, str) or not name:
            raise PreparationError("every testbed node needs trace_node")
        if name in mapping:
            raise PreparationError(f"testbed maps {name} more than once")
        if not isinstance(node.get("host"), str) or not node["host"].strip():
            raise PreparationError(f"testbed node {name} needs its real physical host")
        address = node.get("node_daemon_address", node.get("agent_address"))
        if not isinstance(address, str) or not address.strip():
            raise PreparationError(f"testbed node {name} needs node_daemon_address")
        if not testbed.get("local", False) and not node.get("ssh"):
            raise PreparationError(f"testbed node {name} needs ssh unless local=true")
        interface = node.get("network_interface")
        if not isinstance(interface, str) or not interface.strip():
            raise PreparationError(
                f"testbed node {name} needs an explicit network_interface "
                "for trace bandwidth replay"
            )
        for field in ("host", "ssh", "node_daemon_address", "network_interface"):
            if "REPLACE_" in str(node.get(field, "")):
                raise PreparationError(
                    f"testbed node {name}.{field} still contains a template placeholder"
                )
        mapping[name] = node
    missing = sorted(expected - mapping.keys())
    if missing:
        raise PreparationError(
            "testbed is missing physical host mappings for trace nodes: " + ", ".join(missing)
        )
    return mapping


def write_testbed_template(archive_path: Path, trace: str, output: Path) -> dict:
    archive_path = archive_path.expanduser().resolve(strict=True)
    output = output.expanduser().absolute()
    if output.exists() or output.is_symlink():
        raise PreparationError(f"refusing to overwrite testbed template: {output}")
    with zipfile.ZipFile(archive_path) as archive:
        members = safe_members(archive)
        traces = discover_traces(members)
        if trace not in traces:
            raise PreparationError(f"unknown trace {trace!r}; available: {', '.join(traces)}")
        name = f"{trace}/nodes.json"
        if name not in members:
            raise PreparationError(f"trace {trace!r} is incomplete; missing: nodes.json")
        with archive.open(members[name]) as source:
            nodes = json.load(source)
    validate_inventory(nodes)
    node_mappings = []
    for node in sorted(nodes, key=lambda item: item["node_id"]):
        placeholder = "REPLACE_" + re.sub(r"[^A-Z0-9]", "_", node["node_id"].upper()) + "_IP"
        node_mappings.append(
            {
                "trace_node": node["node_id"],
                "host": placeholder,
                "ssh": "USER@" + placeholder,
                "node_daemon_address": placeholder + ":50052",
                "network_interface": "REPLACE_TEST_NIC",
            }
        )
    template = {
        "_template": True,
        "_template_instructions": (
            "Fill all physical host, SSH user and dedicated test network interface "
            "placeholders; then set _template=false."
        ),
        "coordinator": {
            "host": "REPLACE_COORDINATOR_IP",
            "ssh": "USER@REPLACE_COORDINATOR_IP",
            "address": "REPLACE_COORDINATOR_IP:50051",
        },
        "nodes": node_mappings,
        "paths": {
            "reclaimnet": "/opt/reclaimnet",
            "job_binary": "/opt/rnbt/bin/rnbt-job",
            "load_binary": "/opt/rnbt/bin/rnbt-load",
            "state_root": "/var/lib/rnbt/state",
            "log_dir": "/var/lib/rnbt/logs",
            "checkpoint_dir": "/var/lib/rnbt/checkpoints",
            "coordinator_data": "/var/lib/rnbt/coordinator",
        },
        "local": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("x", encoding="utf-8") as target:
        json.dump(template, target, ensure_ascii=False, indent=2)
        target.write("\n")
    return template


def window_samples(samples: list, node: str, kind: str, start: int, end: int) -> tuple[list, dict]:
    samples.sort(key=lambda value: value[0])
    previous = None
    selected = []
    for timestamp, sample in samples:
        if timestamp <= start:
            previous = (timestamp, sample)
        elif timestamp < end:
            selected.append((timestamp, sample))
    if previous:
        initial = previous
        source = "at_window_start" if initial[0] == start else "inherited_before_window"
    elif selected:
        initial = selected[0]
        source = "first_after_window_start"
    else:
        raise PreparationError(
            f"node {node} has no {kind} at or before the end of the selected window"
        )
    output_samples = [{"offset_s": 0.0, **initial[1]}]
    for timestamp, sample in selected:
        offset = (timestamp - start) / NS

        if output_samples[-1]["offset_s"] == offset:
            output_samples[-1] = {"offset_s": offset, **sample}
        else:
            output_samples.append({"offset_s": offset, **sample})
    metadata = {
        "interpolation": "zero_order_hold",
        "rates_unit": "bits_per_second",
        "initial_sample_source": source,
        "initial_sample_timestamp_ns": initial[0],
        "initial_sample_backfill_s": max(0, initial[0] - start) / NS,
        "note": (
            "The first recorded sample is applied from offset 0; no zero-value sample was invented."
            if source == "first_after_window_start"
            else "Start value uses the latest observation at or before the window."
        ),
    }
    return output_samples, metadata


def make_profiles(telemetry: Path, profiles: Path, nodes: list, start: int, end: int) -> list[dict]:
    by_node = {node["node_id"]: [] for node in nodes}
    for record in records(telemetry):
        node = record.get("node_id")
        if node not in by_node:
            raise PreparationError(f"telemetry references unknown node {node!r}")
        timestamp = checked_timestamp(record.get("timestamp"), "telemetry.timestamp")
        sample = {
            "gpu_util_pct": checked_number(record.get("gpu_util_pct"), "gpu_util_pct", maximum=100),
            "link_tx_bps": checked_number(record.get("link_tx_bps"), "link_tx_bps"),
            "link_rx_bps": checked_number(record.get("link_rx_bps"), "link_rx_bps"),
        }
        by_node[node].append((timestamp, sample))
    summaries = []
    profiles.mkdir()
    for node, samples in sorted(by_node.items()):
        output_samples, metadata = window_samples(samples, node, "telemetry", start, end)
        write_json(
            profiles / f"{node}.json",
            {
                "schema_version": 1,
                "trace_node": node,
                "window_start_ns": start,
                "window_end_ns": end,
                "samples": output_samples,
                "metadata": metadata,
            },
        )
        summaries.append({"trace_node": node, "sample_count": len(output_samples), **metadata})
    return summaries


def make_capacity_profiles(
    capacity: Path, profiles: Path, nodes: list, start: int, end: int
) -> list[dict]:
    by_node = {node["node_id"]: [] for node in nodes}
    for record in records(capacity):
        node = record.get("node_id")
        if node not in by_node:
            raise PreparationError(f"bandwidth_samples references unknown node {node!r}")
        timestamp = checked_timestamp(record.get("timestamp"), "bandwidth_samples.timestamp")
        rate = checked_number(record.get("bps"), "bandwidth_samples.bps")
        by_node[node].append((timestamp, {"bps": rate}))
    summaries = []
    for node, samples in sorted(by_node.items()):
        output_samples, metadata = window_samples(samples, node, "bandwidth samples", start, end)
        write_json(
            profiles / f"{node}.capacity.json",
            {
                "schema_version": 1,
                "trace_node": node,
                "window_start_ns": start,
                "window_end_ns": end,
                "samples": output_samples,
                "metadata": metadata,
            },
        )
        summaries.append({"trace_node": node, "sample_count": len(output_samples), **metadata})
    return summaries


def resource_summary(
    nodes: list, jobs: Path, segments: Path, mapping: dict, start: int, end: int
) -> dict:
    job_rows = list(records(jobs))
    segment_rows = list(records(segments))
    active_segments = [
        row
        for row in segment_rows
        if checked_timestamp(row.get("start_ns"), "job_segments.start_ns") < end
        and checked_timestamp(row.get("end_ns"), "job_segments.end_ns") > start
    ]
    active_ids = {row.get("job_id") for row in active_segments}
    carry_in = [row for row in active_segments if row["start_ns"] < start]
    return {
        "trace_node_count": len(nodes),
        "trace_gpu_count": sum(node["gpu_count"] for node in nodes),
        "trace_inventory_vram_bytes": sum(node["total_vram_bytes"] for node in nodes),
        "trace_job_count": len(job_rows),
        "overlapping_segment_count": len(active_segments),
        "overlapping_job_count": len(active_ids),
        "carry_in_segment_count": len(carry_in),
        "all_job_and_reclaim_history_retained": True,
        "physical_capacity_verified": False,
        "physical_capacity_note": (
            "Inventory is the trace requirement; CUDA/NVML must verify actual hardware "
            "at runtime."
        ),
        "nodes": [
            {
                "trace_node": node["node_id"],
                "host": mapping[node["node_id"]]["host"],
                "network_interface": mapping[node["node_id"]]["network_interface"],
                "gpu_count": node["gpu_count"],
                "gpu_model": node.get("gpu_model"),
                "total_vram_bytes": node["total_vram_bytes"],
                "link_capacity_bps": node.get("link_capacity_bps", node.get("link_cap_bps")),
            }
            for node in nodes
        ],
    }


def cache_directory(value: Path | None, output: Path) -> Path:
    project = Path(__file__).resolve().parent.parent
    if value is None:
        candidate = (
            Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache")))
            / "rnbt"
            / "trace-data"
        )
        value = (
            candidate
            if not candidate.expanduser().resolve().is_relative_to(project)
            else Path.home() / ".cache/rnbt/trace-data"
        )
    cache = value.expanduser().resolve()
    if cache.is_relative_to(project):
        raise PreparationError(f"trace cache must be outside the project directory: {cache}")
    if cache.is_relative_to(output) or output.is_relative_to(cache):
        raise PreparationError("trace cache and output must not contain each other")
    return cache


@contextmanager
def cache_lock(root: Path, directory: Path):
    if not directory.resolve().is_relative_to(root):
        raise PreparationError(f"cache path escapes its root: {directory}")
    directory.mkdir(parents=True, exist_ok=True)
    lock = directory / ".prepare.lock"
    if lock.is_symlink():
        raise PreparationError(f"invalid cache lock: {lock}")
    with lock.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield


def load_cache(directory: Path, identity: dict, names: set[str]) -> dict | None:
    if directory.is_symlink():
        raise PreparationError(f"invalid cache directory: {directory}")
    if not directory.exists():
        return None
    try:
        if not directory.is_dir() or {item.name for item in directory.iterdir()} != names | {
            ".cache.json"
        }:
            raise ValueError("unexpected or missing cached files")
        metadata_path = directory / ".cache.json"
        if metadata_path.is_symlink():
            raise ValueError("metadata is a symbolic link")
        metadata = read_json(metadata_path)
        if not isinstance(metadata, dict) or any(
            metadata.get(key) != value for key, value in identity.items()
        ):
            raise ValueError("cache identity does not match archive, trace or window")
        hashes = metadata.get("files")
        if not isinstance(hashes, dict) or set(hashes) != names:
            raise ValueError("invalid cached file inventory")
        for name in sorted(names):
            path = directory / name
            if path.is_symlink() or not path.is_file() or file_digest(path) != hashes[name]:
                raise ValueError(f"cached file hash mismatch: {name}")
        return metadata
    except (OSError, ValueError, TypeError) as error:
        raise PreparationError(f"corrupt trace cache {directory}: {error}") from error


def publish_cache(directory: Path, build) -> dict:
    directory.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{directory.name}.prepare-", dir=directory.parent
    ) as temporary:
        stage = Path(temporary) / "entry"
        metadata = build(stage)
        write_json(stage / ".cache.json", metadata)
        os.replace(stage, directory)
    return metadata


def prepare(
    archive_path: Path,
    trace: str,
    from_time: str,
    to_time: str,
    testbed_path: Path,
    output: Path,
    cache_dir: Path | None = None,
) -> dict:
    archive_path = archive_path.expanduser().resolve(strict=True)
    testbed_path = testbed_path.expanduser().resolve(strict=True)
    output = output.expanduser().absolute()
    if output.is_symlink() or (output.exists() and (not output.is_dir() or any(output.iterdir()))):
        raise PreparationError(f"output must be a new or empty directory: {output}")
    output = output.resolve()
    cache = cache_directory(cache_dir, output)
    archive_digest = file_digest(archive_path)
    with zipfile.ZipFile(archive_path) as archive:
        members = safe_members(archive)
        traces = discover_traces(members)
        if trace not in traces:
            raise PreparationError(f"unknown trace {trace!r}; available: {', '.join(traces)}")
        absent = [name for name in INPUT_FILES if f"{trace}/{name}" not in members]
        if absent:
            raise PreparationError(f"trace {trace!r} is incomplete; missing: {', '.join(absent)}")
        files = list(INPUT_FILES) + [
            name for name in OPTIONAL_FILES if f"{trace}/{name}" in members
        ]
        sizes = [members[f"{trace}/{name}"].file_size for name in files]
        if any(size > 128 * 1024**2 for size in sizes) or sum(sizes) > 512 * 1024**2:
            raise PreparationError("selected trace inputs exceed the safe extraction size limit")
        with archive.open(members[f"{trace}/manifest.json"]) as source:
            manifest = json.load(source)
        if not isinstance(manifest, dict):
            raise PreparationError("manifest.json must be an object")
        origin, offset = parse_iso(manifest["window_start"])
        limit, _ = parse_iso(manifest["window_end"])
        start, end = parse_bound(from_time, origin, offset), parse_bound(to_time, origin, offset)
        if not origin <= start < end <= limit:
            raise PreparationError(
                "selected [from,to) window must be non-empty and inside manifest "
                f"[{manifest['window_start']}, {manifest['window_end']})"
            )
        testbed = read_json(testbed_path)
        with archive.open(members[f"{trace}/nodes.json"]) as source:
            nodes = json.load(source)
        mapping = validate_mapping(nodes, testbed)
        trace_cache = cache / "v1" / archive_digest["sha256"] / trace
        input_dir = trace_cache / "input_trace"
        profile_dir = trace_cache / "windows" / f"{start}_{end}"
        identity = {"schema_version": 1, "archive_sha256": archive_digest["sha256"], "trace": trace}
        input_identity = {**identity, "kind": "input_trace"}
        profile_identity = {
            **identity,
            "kind": "window_profiles",
            "window_start_ns": start,
            "window_end_ns": end,
        }
        profile_names = {
            f"{node['node_id']}{suffix}.json" for node in nodes for suffix in ("", ".capacity")
        }
        with cache_lock(cache, trace_cache):
            inputs = load_cache(input_dir, input_identity, set(files))
            if inputs is None:

                def build_inputs(stage):
                    stage.mkdir()
                    hashes = {}
                    for name in files:
                        with (
                            archive.open(members[f"{trace}/{name}"]) as source,
                            (stage / name).open("wb") as target,
                        ):
                            shutil.copyfileobj(source, target, 1024 * 1024)
                        hashes[name] = file_digest(stage / name)
                        declared = manifest.get("files", {}).get(name, {})
                        if declared.get("sha256") and declared["sha256"] != hashes[name]["sha256"]:
                            raise PreparationError(f"manifest SHA256 mismatch for {name}")
                        if "bytes" in declared and declared["bytes"] != hashes[name]["bytes"]:
                            raise PreparationError(f"manifest byte count mismatch for {name}")
                    return {**input_identity, "files": hashes}

                inputs = publish_cache(input_dir, build_inputs)

            for name, digest in inputs["files"].items():
                declared = manifest.get("files", {}).get(name, {})
                if (
                    declared.get("sha256")
                    and declared["sha256"] != digest["sha256"]
                    or "bytes" in declared
                    and declared["bytes"] != digest["bytes"]
                ):
                    raise PreparationError(f"corrupt trace cache: manifest mismatch for {name}")
            if not profile_dir.resolve().is_relative_to(trace_cache.resolve()):
                raise PreparationError(f"cache profile path escapes its trace: {profile_dir}")
            profiles = load_cache(profile_dir, profile_identity, profile_names)
            if profiles is None:

                def build_profiles(stage):
                    gpu = make_profiles(input_dir / "telemetry.jsonl", stage, nodes, start, end)
                    capacity = make_capacity_profiles(
                        input_dir / "bandwidth_samples.jsonl", stage, nodes, start, end
                    )
                    return {
                        **profile_identity,
                        "files": {
                            name: file_digest(stage / name) for name in sorted(profile_names)
                        },
                        "profiles": gpu,
                        "capacity_profiles": capacity,
                    }

                profiles = publish_cache(profile_dir, build_profiles)
            if not isinstance(profiles.get("profiles"), list) or not isinstance(
                profiles.get("capacity_profiles"), list
            ):
                raise PreparationError(
                    f"corrupt trace cache: missing profile summaries in {profile_dir}"
                )
        summary = resource_summary(
            nodes, input_dir / "jobs.jsonl", input_dir / "job_segments.jsonl", mapping, start, end
        )
        window = {"from": format_iso(start, offset), "to": format_iso(end, offset)}
        result = {
            "schema_version": 1,
            "used_trace": trace,
            "source_archive": str(archive_path),
            "archive_digest": archive_digest,
            "source_run_id": manifest.get("run_id"),
            "trace_directory": str(input_dir),
            "profile_directory": str(profile_dir),
            "cache_directory": str(cache),
            "source_window": {"from": manifest["window_start"], "to": manifest["window_end"]},
            "window": window,
            "window_start_ns": start,
            "window_end_ns": end,
            "duration_s": (end - start) / NS,
            "input_hashes": inputs["files"],
            "testbed": str(testbed_path),
            "testbed_digest": file_digest(testbed_path),
            "resource_summary": summary,
            "profiles": profiles["profiles"],
            "capacity_profiles": profiles["capacity_profiles"],
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix=f".{output.name}.prepare-", dir=output.parent
        ) as temporary:
            stage = Path(temporary) / "prepared"
            stage.mkdir()
            write_json(
                stage / "replay.json",
                {
                    "run_id": (
                        f"{re.sub(r'[^A-Za-z0-9_.-]', '_', output.name)}"
                        f"-{uuid.uuid4().hex[:12]}"
                    ),
                    "trace": str(input_dir),
                    "testbed": str(testbed_path),
                    "mode": "jobs",
                    "window": window,
                    "workload_profiles": str(profile_dir),
                    "replay_link_capacity": True,
                    "timing": {"compress_idle": False, "jobs_warmup_s": 0, "drain_s": 0},
                    "output": str(output),
                },
            )
            write_json(stage / "preparation.json", result)
            if output.exists():
                output.rmdir()
            os.replace(stage, output)
    return result


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description=(
            "Prepare one archive trace and a bounded physical-GPU replay configuration.\n"
            "\n"
            "Only input files consumed by replay are extracted. Recorded outcomes, databases,\n"
            "and checkpoint metadata are deliberately not used as live execution results.\n"
            "Trace inputs and generated window profiles stay in an external shared cache;\n"
            "run directories contain path references rather than copied trace data.\n"
        )
    )
    result.add_argument("--archive", required=True, type=Path)
    result.add_argument(
        "--list-traces", action="store_true", help="list available traces and manifest windows"
    )
    result.add_argument("--trace", help="one archive trace, for example trace_1 or trace2")
    result.add_argument(
        "--write-testbed-template",
        type=Path,
        help="write a fill-in physical-host template for --trace; no window or testbed is required",
    )
    result.add_argument(
        "--from",
        dest="from_time",
        help="inclusive ISO-8601 time or local day:N (day:0 is the origin date)",
    )
    result.add_argument("--to", dest="to_time", help="exclusive ISO-8601 time or local day:N")
    result.add_argument("--testbed", type=Path, help="existing physical-host testbed JSON")
    result.add_argument(
        "--output", type=Path, help="new or empty replay run directory (configuration only)"
    )
    result.add_argument(
        "--cache-dir",
        type=Path,
        help=(
            "shared trace/profile cache outside the project "
            "(default: $XDG_CACHE_HOME/rnbt/trace-data or ~/.cache/rnbt/trace-data)"
        ),
    )
    return result


def main(argv=None) -> int:
    cli = parser()
    args = cli.parse_args(argv)
    try:
        if args.list_traces:
            if args.write_testbed_template:
                cli.error("--list-traces and --write-testbed-template cannot be combined")
            with zipfile.ZipFile(args.archive.expanduser()) as archive:
                members = safe_members(archive)
                for trace in discover_traces(members):
                    with archive.open(members[f"{trace}/manifest.json"]) as source:
                        manifest = json.load(source)
                    print(f"{trace}\t{manifest['window_start']}\t{manifest['window_end']}")
            return 0
        if args.write_testbed_template:
            if args.trace is None:
                cli.error("--trace is required with --write-testbed-template")
            template = write_testbed_template(args.archive, args.trace, args.write_testbed_template)
            print(
                f"Wrote physical-host template for {len(template['nodes'])} trace nodes: "
                f"{args.write_testbed_template.expanduser().resolve()}"
            )
            print(
                "Fill physical hosts, SSH users and dedicated test network interfaces, "
                "then set _template=false."
            )
            return 0
        for name in ("trace", "from_time", "to_time", "testbed", "output"):
            if getattr(args, name) is None:
                cli.error(
                    f"--{name.removesuffix('_time')} is required unless --list-traces is used"
                )
        result = prepare(
            args.archive,
            args.trace,
            args.from_time,
            args.to_time,
            args.testbed,
            args.output,
            args.cache_dir,
        )
        summary = result["resource_summary"]
        print(
            f"Prepared {result['used_trace']}: "
            f"[{result['window']['from']}, {result['window']['to']})"
        )
        print(
            f"Required inventory: {summary['trace_node_count']} nodes, "
            f"{summary['trace_gpu_count']} GPUs; "
            f"{summary['carry_in_segment_count']} carry-in segments"
        )
        print("Replay configuration: " f"{args.output.expanduser().resolve() / 'replay.json'}")
        print(
            "Input hashes and resource requirements: "
            f"{args.output.expanduser().resolve() / 'preparation.json'}"
        )
        print(f"External trace cache: {result['cache_directory']}")
        return 0
    except (
        PreparationError,
        OSError,
        ValueError,
        TypeError,
        KeyError,
        zipfile.BadZipFile,
        RuntimeError,
    ) as error:
        print(f"prepare_trace: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
