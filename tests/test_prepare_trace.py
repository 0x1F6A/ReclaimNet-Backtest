import contextlib
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import io
import json
import os
from pathlib import Path
import stat
import tempfile
import unittest
from unittest import mock
import zipfile


MODULE = Path(__file__).resolve().parents[1] / "scripts" / "prepare_trace.py"
SPEC = importlib.util.spec_from_file_location("prepare_trace", MODULE)
prepare_trace = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prepare_trace)


class PrepareTraceTests(unittest.TestCase):
    def test_fractional_and_scientific_day_bounds(self):
        origin, offset = prepare_trace.parse_iso("2026-03-01T12:00:00+08:00")
        midnight = prepare_trace.parse_iso("2026-03-01T00:00:00+08:00")[0]
        self.assertEqual(
            prepare_trace.parse_bound("day:1.5", origin, offset),
            midnight + 3 * prepare_trace.DAY_NS // 2,
        )
        self.assertEqual(
            prepare_trace.parse_bound("day:5e-1", origin, offset),
            midnight + prepare_trace.DAY_NS // 2,
        )
        for bound in ("day:NaN", "day:inf", "day:1e99999"):
            with self.assertRaises(prepare_trace.PreparationError):
                prepare_trace.parse_bound(bound, origin, offset)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.archive = self.root / "traces.zip"
        self.testbed = self.root / "testbed.json"
        self.output = self.root / "run"
        self.cache = self.root / "external-data"
        self.origin = prepare_trace.parse_iso("2026-03-01T00:00:00+08:00")[0]
        self.day = prepare_trace.DAY_NS
        self.node = {
            "node_id": "node-001",
            "gpu_count": 2,
            "total_vram_bytes": 48 * 1024**3,
            "gpu_model": "physical-device",
            "link_capacity_bps": 1_000_000_000,
        }
        self.mapping = {
            "nodes": [
                {
                    "trace_node": "node-001",
                    "host": "gpu01",
                    "ssh": "user@gpu01",
                    "node_daemon_address": "gpu01:50052",
                    "network_interface": "ens6",
                }
            ]
        }
        self.testbed.write_text(json.dumps(self.mapping))
        self.telemetry = [
            self.sample(self.origin - 10, 12),
            self.sample(self.origin + self.day // 2, 34),
            self.sample(self.origin + self.day, 56),
            self.sample(self.origin + 2 * self.day, 78),
        ]
        self.files = {
            "manifest.json": json.dumps(
                {
                    "window_start": "2026-03-01T00:00:00+08:00",
                    "window_end": "2026-03-04T00:00:00+08:00",
                    "run_id": "fixture",
                }
            ),
            "nodes.json": json.dumps([self.node]),
            "topology.json": json.dumps({"links": []}),
            "jobs.jsonl": self.jsonl(
                [
                    {
                        "job_id": "carry-in",
                        "submit_time_ns": self.origin - 100,
                        "requested_runtime_s": 600,
                    }
                ]
            ),
            "reclaims.jsonl": self.jsonl(
                [{"timestamp": self.origin - 1000}, {"timestamp": self.origin + self.day}]
            ),
            "job_segments.jsonl": self.jsonl(
                [
                    {
                        "job_id": "carry-in",
                        "start_ns": self.origin - 100,
                        "end_ns": self.origin + self.day,
                    },
                    {
                        "job_id": "ended-at-from",
                        "start_ns": self.origin - 1000,
                        "end_ns": self.origin,
                    },
                    {
                        "job_id": "starts-at-to",
                        "start_ns": self.origin + self.day,
                        "end_ns": self.origin + 2 * self.day,
                    },
                ]
            ),
            "telemetry.jsonl": self.jsonl(self.telemetry),
            "bandwidth_samples.jsonl": self.jsonl(
                [{"node_id": "node-001", "bps": 50, "timestamp": self.origin}]
            ),
            "core_bandwidth_samples.jsonl": "",
            "hazard_samples.jsonl": "",
            "historical_migration_profiles.jsonl": "",
            "schema.json": "{}",
            "reclaimnet.db": "DO NOT EXTRACT",
            "p2_candidates.jsonl": "DO NOT EXTRACT",
        }
        self.make_archive()

    @staticmethod
    def jsonl(rows):
        return "".join(json.dumps(row) + "\n" for row in rows)

    @staticmethod
    def sample(timestamp, util):
        return {
            "node_id": "node-001",
            "timestamp": timestamp,
            "gpu_util_pct": util,
            "link_tx_bps": util * 1_000_000,
            "link_rx_bps": util * 100_000,
        }

    def make_archive(self, traces=("trace_1", "trace2", "trace3", "trace4", "trace5"), extra=None):
        with zipfile.ZipFile(self.archive, "w", zipfile.ZIP_DEFLATED) as archive:
            for trace in traces:
                for name, value in self.files.items():
                    archive.writestr(f"{trace}/{name}", value)
            for name, value in (extra or {}).items():
                archive.writestr(name, value)

    def prepare(self, start="day:0", end="day:1", trace="trace_1", output=None):
        return prepare_trace.prepare(
            self.archive,
            trace,
            start,
            end,
            self.testbed,
            output or self.output,
            cache_dir=self.cache,
        )

    def config(self, output=None):
        return json.loads(((output or self.output) / "replay.json").read_text())

    def profile_directory(self):
        return Path(self.config()["workload_profiles"])

    def profile(self):
        return json.loads((self.profile_directory() / "node-001.json").read_text())

    def test_lists_all_five_traces_without_other_arguments(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(
                prepare_trace.main(["--archive", str(self.archive), "--list-traces"]), 0
            )
        self.assertEqual(
            {line.split("\t")[0] for line in output.getvalue().splitlines()},
            {"trace_1", "trace2", "trace3", "trace4", "trace5"},
        )
        self.assertFalse(self.output.exists())

    def test_write_template_needs_no_window_or_testbed(self):
        template_path = self.root / "config" / "template.json"
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            status = prepare_trace.main(
                [
                    "--archive",
                    str(self.archive),
                    "--trace",
                    "trace_1",
                    "--write-testbed-template",
                    str(template_path),
                ]
            )
        self.assertEqual(status, 0)
        template = json.loads(template_path.read_text())
        self.assertTrue(template["_template"])
        self.assertEqual(template["nodes"][0]["host"], "REPLACE_NODE_001_IP")
        self.assertEqual(template["nodes"][0]["ssh"], "USER@REPLACE_NODE_001_IP")
        self.assertEqual(template["nodes"][0]["node_daemon_address"], "REPLACE_NODE_001_IP:50052")
        self.assertEqual(template["nodes"][0]["network_interface"], "REPLACE_TEST_NIC")
        self.assertEqual(template["coordinator"]["address"], "REPLACE_COORDINATOR_IP:50051")
        self.assertEqual(template["paths"]["load_binary"], "/opt/rnbt/bin/rnbt-load")
        self.assertFalse(self.output.exists())

    def test_template_never_overwrites_existing_testbed(self):
        original = self.testbed.read_text()
        with self.assertRaisesRegex(prepare_trace.PreparationError, "refusing to overwrite"):
            prepare_trace.write_testbed_template(self.archive, "trace_1", self.testbed)
        self.assertEqual(self.testbed.read_text(), original)

    def test_unfilled_template_is_rejected_even_with_flag_removed(self):
        template_path = self.root / "template.json"
        template = prepare_trace.write_testbed_template(self.archive, "trace_1", template_path)
        self.testbed.write_text(json.dumps(template))
        with self.assertRaisesRegex(prepare_trace.PreparationError, "unfilled template"):
            self.prepare()
        template["_template"] = False
        self.testbed.write_text(json.dumps(template))
        with self.assertRaisesRegex(prepare_trace.PreparationError, "template placeholder"):
            self.prepare()

    def test_selected_trace_full_history_and_exclusive_end(self):
        result = self.prepare(trace="trace3")
        config = json.loads((self.output / "replay.json").read_text())
        self.assertEqual(result["used_trace"], "trace3")
        self.assertEqual(config["mode"], "jobs")
        self.assertEqual(config["testbed"], str(self.testbed))
        trace_directory = Path(config["trace"])
        profile_directory = Path(config["workload_profiles"])
        self.assertTrue(trace_directory.is_absolute())
        self.assertTrue(profile_directory.is_absolute())
        self.assertTrue(trace_directory.is_relative_to(self.cache))
        self.assertTrue(profile_directory.is_relative_to(self.cache))
        self.assertEqual(result["trace_directory"], str(trace_directory))
        self.assertEqual(result["profile_directory"], str(profile_directory))
        self.assertEqual(result["cache_directory"], str(self.cache))
        self.assertTrue(config["replay_link_capacity"])
        self.assertEqual(config["output"], str(self.output))
        self.assertEqual(config["window"]["to"], "2026-03-02T00:00:00.000000000+08:00")
        self.assertFalse((trace_directory / "reclaimnet.db").exists())
        self.assertFalse((trace_directory / "p2_candidates.jsonl").exists())
        self.assertEqual((trace_directory / "jobs.jsonl").read_text(), self.files["jobs.jsonl"])
        self.assertEqual(
            (trace_directory / "reclaims.jsonl").read_text(), self.files["reclaims.jsonl"]
        )
        self.assertEqual(
            set(result["input_hashes"]),
            set(prepare_trace.INPUT_FILES + prepare_trace.OPTIONAL_FILES),
        )
        self.assertEqual(len(result["archive_digest"]["sha256"]), 64)
        summary = result["resource_summary"]
        self.assertEqual(summary["trace_gpu_count"], 2)
        self.assertEqual(summary["overlapping_segment_count"], 1)
        self.assertEqual(summary["carry_in_segment_count"], 1)
        profile = self.profile()
        self.assertEqual(
            profile["samples"],
            [
                {
                    "offset_s": 0.0,
                    "gpu_util_pct": 12,
                    "link_tx_bps": 12_000_000,
                    "link_rx_bps": 1_200_000,
                },
                {
                    "offset_s": 43200.0,
                    "gpu_util_pct": 34,
                    "link_tx_bps": 34_000_000,
                    "link_rx_bps": 3_400_000,
                },
            ],
        )
        self.assertEqual(profile["metadata"]["initial_sample_source"], "inherited_before_window")

    def test_iso_offsets_and_nanoseconds_are_exact(self):
        self.prepare("2026-02-28T16:00:00.000000001Z", "2026-03-01T16:00:00.000000001Z")
        profile = self.profile()
        self.assertEqual(profile["window_start_ns"], self.origin + 1)
        self.assertEqual(profile["window_end_ns"], self.origin + self.day + 1)
        self.assertEqual([sample["gpu_util_pct"] for sample in profile["samples"]], [12, 34, 56])

    def test_first_observation_backfills_start_with_explicit_metadata(self):
        self.files["telemetry.jsonl"] = self.jsonl(self.telemetry[1:])
        self.make_archive()
        self.prepare()
        profile = self.profile()
        self.assertEqual(profile["samples"][0]["gpu_util_pct"], 34)
        self.assertEqual(profile["samples"][0]["offset_s"], 0)
        self.assertEqual(profile["metadata"]["initial_sample_source"], "first_after_window_start")
        self.assertEqual(profile["metadata"]["initial_sample_backfill_s"], 43200)

    def test_exact_start_observation_is_applied(self):
        self.files["telemetry.jsonl"] = self.jsonl([self.sample(self.origin, 21)])
        self.make_archive()
        self.prepare()
        self.assertEqual(self.profile()["metadata"]["initial_sample_source"], "at_window_start")
        self.assertEqual(len(self.profile()["samples"]), 1)

    def test_day_index_uses_origin_local_midnight(self):
        origin, offset = prepare_trace.parse_iso("2026-03-01T12:34:56+08:00")
        self.assertEqual(prepare_trace.parse_bound("day:0", origin, offset), self.origin)
        self.assertEqual(prepare_trace.parse_bound("day:1", origin, offset), self.origin + self.day)

    def test_invalid_or_outside_windows_fail_without_output(self):
        windows = [
            ("day:-1", "day:1"),
            ("day:1", "day:1"),
            ("day:2", "day:1"),
            ("day:0", "day:4"),
            ("day:NaN", "day:1"),
            ("2026-02-30T00:00:00+08:00", "day:1"),
            ("2026-03-01T00:00:00+25:00", "day:1"),
            ("2026-03-01T00:00:00.1234567891Z", "day:1"),
        ]
        for start, end in windows:
            with (
                self.subTest(start=start, end=end),
                self.assertRaises(prepare_trace.PreparationError),
            ):
                self.prepare(start, end)
            self.assertFalse(self.output.exists())

    def test_unsafe_zip_members_are_rejected_even_when_not_extracted(self):
        for name in (
            "../escape",
            "/absolute",
            "trace_1/../../escape",
            "trace_1\\escape",
            "C:/escape",
        ):
            with self.subTest(name=name):
                self.make_archive(extra={name: "bad"})
                with self.assertRaisesRegex(prepare_trace.PreparationError, "unsafe ZIP"):
                    self.prepare()
                self.assertFalse(self.output.exists())

    def test_zip_symlink_is_rejected(self):
        info = zipfile.ZipInfo("trace_1/escape")
        info.create_system = 3
        info.external_attr = (stat.S_IFLNK | 0o777) << 16
        with zipfile.ZipFile(self.archive, "a") as archive:
            archive.writestr(info, "../../escape")
        with self.assertRaisesRegex(prepare_trace.PreparationError, "symbolic links"):
            self.prepare()

    def test_incomplete_trace_rejected(self):
        del self.files["job_segments.jsonl"]
        self.make_archive()
        with self.assertRaisesRegex(prepare_trace.PreparationError, "missing: job_segments.jsonl"):
            self.prepare()

    def test_unknown_trace_rejected(self):
        with self.assertRaisesRegex(prepare_trace.PreparationError, "unknown trace"):
            self.prepare(trace="trace6")

    def test_missing_physical_mapping_rejected(self):
        self.testbed.write_text(json.dumps({"nodes": []}))
        with self.assertRaisesRegex(
            prepare_trace.PreparationError, "missing physical host mappings.*node-001"
        ):
            self.prepare()
        self.assertFalse(self.output.exists())

    def test_duplicate_mapping_rejected(self):
        self.mapping["nodes"].append(self.mapping["nodes"][0])
        self.testbed.write_text(json.dumps(self.mapping))
        with self.assertRaisesRegex(prepare_trace.PreparationError, "more than once"):
            self.prepare()

    def test_missing_host_rejected(self):
        del self.mapping["nodes"][0]["host"]
        self.testbed.write_text(json.dumps(self.mapping))
        with self.assertRaisesRegex(prepare_trace.PreparationError, "real physical host"):
            self.prepare()

    def test_missing_interface_rejected_without_guessing_trace_iface(self):
        del self.mapping["nodes"][0]["network_interface"]
        self.testbed.write_text(json.dumps(self.mapping))
        self.node["iface"] = "eth0"
        self.files["nodes.json"] = json.dumps([self.node])
        self.make_archive()
        with self.assertRaisesRegex(prepare_trace.PreparationError, "explicit network_interface"):
            self.prepare()

    def test_capacity_zero_and_independent_timepoints_are_preserved(self):
        self.files["bandwidth_samples.jsonl"] = self.jsonl(
            [
                {"node_id": "node-001", "timestamp": self.origin - 1, "bps": 400_000_000},
                {
                    "node_id": "node-001",
                    "timestamp": self.origin + 123 * prepare_trace.NS,
                    "bps": 0,
                },
                {
                    "node_id": "node-001",
                    "timestamp": self.origin + 456 * prepare_trace.NS,
                    "bps": 50_000_000,
                },
                {"node_id": "node-001", "timestamp": self.origin + self.day, "bps": 1},
            ]
        )
        self.make_archive()
        result = self.prepare()
        capacity = json.loads((self.profile_directory() / "node-001.capacity.json").read_text())
        self.assertEqual(
            capacity["samples"],
            [
                {"offset_s": 0, "bps": 400_000_000},
                {"offset_s": 123, "bps": 0},
                {"offset_s": 456, "bps": 50_000_000},
            ],
        )
        self.assertEqual(capacity["metadata"]["initial_sample_source"], "inherited_before_window")
        self.assertEqual(result["capacity_profiles"][0]["sample_count"], 3)
        self.assertEqual(result["resource_summary"]["nodes"][0]["network_interface"], "ens6")

    def test_capacity_first_observation_backfill_is_explicit(self):
        self.files["bandwidth_samples.jsonl"] = self.jsonl(
            [
                {"node_id": "node-001", "timestamp": self.origin + 30 * prepare_trace.NS, "bps": 0},
            ]
        )
        self.make_archive()
        self.prepare()
        capacity = json.loads((self.profile_directory() / "node-001.capacity.json").read_text())
        self.assertEqual(capacity["samples"][0], {"offset_s": 0, "bps": 0})
        self.assertEqual(capacity["metadata"]["initial_sample_source"], "first_after_window_start")
        self.assertEqual(capacity["metadata"]["initial_sample_backfill_s"], 30)

    def test_missing_capacity_samples_rejected(self):
        self.files["bandwidth_samples.jsonl"] = ""
        self.make_archive()
        with self.assertRaisesRegex(prepare_trace.PreparationError, "no bandwidth samples"):
            self.prepare()
        self.assertFalse(self.output.exists())

    def test_missing_telemetry_is_error_without_invented_zero(self):
        self.files["telemetry.jsonl"] = ""
        self.make_archive()
        with self.assertRaisesRegex(prepare_trace.PreparationError, "no telemetry"):
            self.prepare()
        self.assertFalse(self.output.exists())
        self.assertEqual(list(self.root.glob(".run.prepare-*")), [])

    def test_telemetry_after_window_only_is_error(self):
        self.files["telemetry.jsonl"] = self.jsonl([self.sample(self.origin + self.day, 99)])
        self.make_archive()
        with self.assertRaisesRegex(prepare_trace.PreparationError, "no telemetry"):
            self.prepare()

    def test_nan_and_out_of_range_utilization_are_rejected(self):
        for util in (float("nan"), 101, -1, True):
            with self.subTest(util=util):
                self.files["telemetry.jsonl"] = self.jsonl([self.sample(self.origin, util)])
                self.make_archive()
                with self.assertRaises(prepare_trace.PreparationError):
                    self.prepare()

    def test_manifest_integrity_is_verified(self):
        manifest = json.loads(self.files["manifest.json"])
        manifest["files"] = {"nodes.json": {"sha256": "0" * 64}}
        self.files["manifest.json"] = json.dumps(manifest)
        self.make_archive()
        with self.assertRaisesRegex(
            prepare_trace.PreparationError, "SHA256 mismatch for nodes.json"
        ):
            self.prepare()
        self.assertFalse(self.output.exists())

    def test_nonempty_output_is_never_overwritten(self):
        self.output.mkdir()
        sentinel = self.output / "existing.txt"
        sentinel.write_text("keep")
        with self.assertRaisesRegex(prepare_trace.PreparationError, "new or empty"):
            self.prepare()
        self.assertEqual(sentinel.read_text(), "keep")

    def test_empty_output_is_supported(self):
        self.output.mkdir()
        self.prepare()
        self.assertTrue((self.output / "preparation.json").exists())

    def test_output_symlink_is_rejected(self):
        actual = self.root / "actual"
        actual.mkdir()
        self.output.symlink_to(actual, target_is_directory=True)
        with self.assertRaisesRegex(prepare_trace.PreparationError, "new or empty"):
            self.prepare()
        self.assertEqual(list(actual.iterdir()), [])

    def test_output_contains_only_configuration_and_preparation_report(self):
        result = self.prepare()
        self.assertEqual(
            {path.name for path in self.output.iterdir()}, {"replay.json", "preparation.json"}
        )
        self.assertTrue(
            all(path.is_file() and not path.is_symlink() for path in self.output.iterdir())
        )
        self.assertEqual(json.loads((self.output / "preparation.json").read_text()), result)
        self.assertFalse(Path(result["trace_directory"]).is_relative_to(self.output))
        self.assertFalse(Path(result["profile_directory"]).is_relative_to(self.output))

    def test_same_archive_and_window_share_unchanged_cached_files(self):
        first = self.prepare()
        trace_directory = Path(first["trace_directory"])
        profile_directory = Path(first["profile_directory"])
        original_files = {
            path: (path.stat().st_mtime_ns, path.read_bytes())
            for directory in (trace_directory, profile_directory)
            for path in directory.rglob("*")
            if path.is_file()
        }
        second_output = self.root / "second-run"
        second = self.prepare(output=second_output)
        self.assertEqual(first["trace_directory"], second["trace_directory"])
        self.assertEqual(first["profile_directory"], second["profile_directory"])
        self.assertEqual(self.config()["trace"], self.config(second_output)["trace"])
        self.assertEqual(
            self.config()["workload_profiles"], self.config(second_output)["workload_profiles"]
        )
        self.assertTrue(original_files)
        for path, original in original_files.items():
            with self.subTest(path=path):
                self.assertEqual((path.stat().st_mtime_ns, path.read_bytes()), original)

    def test_different_windows_share_trace_but_have_distinct_profiles(self):
        first = self.prepare()
        second = self.prepare("day:1", "day:2", output=self.root / "second-window")
        self.assertEqual(first["trace_directory"], second["trace_directory"])
        self.assertNotEqual(first["profile_directory"], second["profile_directory"])
        first_profile = json.loads((Path(first["profile_directory"]) / "node-001.json").read_text())
        second_profile = json.loads(
            (Path(second["profile_directory"]) / "node-001.json").read_text()
        )
        self.assertEqual(first_profile["window_start_ns"], self.origin)
        self.assertEqual(second_profile["window_start_ns"], self.origin + self.day)
        self.assertEqual(second_profile["samples"][0]["gpu_util_pct"], 56)

    def test_changed_archive_uses_distinct_cached_input_and_profiles(self):
        first = self.prepare()
        first_jobs = Path(first["trace_directory"]) / "jobs.jsonl"
        original_jobs = first_jobs.read_bytes()
        self.files["jobs.jsonl"] += self.jsonl([{"job_id": "new-job"}])
        self.files["telemetry.jsonl"] = self.jsonl([self.sample(self.origin, 91)])
        self.make_archive()
        second = self.prepare(output=self.root / "changed-archive")
        self.assertNotEqual(first["archive_digest"]["sha256"], second["archive_digest"]["sha256"])
        self.assertNotEqual(first["trace_directory"], second["trace_directory"])
        self.assertNotEqual(first["profile_directory"], second["profile_directory"])
        self.assertEqual(first_jobs.read_bytes(), original_jobs)
        self.assertEqual(
            (Path(second["trace_directory"]) / "jobs.jsonl").read_text(), self.files["jobs.jsonl"]
        )
        second_profile = json.loads(
            (Path(second["profile_directory"]) / "node-001.json").read_text()
        )
        self.assertEqual(second_profile["samples"][0]["gpu_util_pct"], 91)

    def test_corrupted_cached_input_is_rejected_without_a_new_output(self):
        first = self.prepare()
        (Path(first["trace_directory"]) / "jobs.jsonl").write_text("corrupted input\n")
        second_output = self.root / "after-corruption"
        with self.assertRaises(prepare_trace.PreparationError):
            self.prepare(output=second_output)
        self.assertFalse(second_output.exists())

    def test_corrupted_cached_profile_is_rejected_without_a_new_output(self):
        first = self.prepare()
        (Path(first["profile_directory"]) / "node-001.json").write_text("{}\n")
        second_output = self.root / "after-corruption"
        with self.assertRaises(prepare_trace.PreparationError):
            self.prepare(output=second_output)
        self.assertFalse(second_output.exists())

    def test_default_cache_uses_external_xdg_directory(self):
        xdg = self.root / "xdg-cache"
        with mock.patch.dict(os.environ, {"XDG_CACHE_HOME": str(xdg)}):
            result = prepare_trace.prepare(
                self.archive, "trace_1", "day:0", "day:1", self.testbed, self.output
            )
        self.assertEqual(Path(result["cache_directory"]), xdg / "rnbt" / "trace-data")
        self.assertFalse(Path(result["cache_directory"]).is_relative_to(MODULE.parents[1]))

    def test_default_cache_uses_home_when_xdg_is_unset(self):
        with (
            mock.patch.dict(os.environ, {}, clear=True),
            mock.patch.object(Path, "home", return_value=self.root),
        ):
            result = prepare_trace.prepare(
                self.archive, "trace_1", "day:0", "day:1", self.testbed, self.output
            )
        self.assertEqual(
            Path(result["cache_directory"]), self.root / ".cache" / "rnbt" / "trace-data"
        )
        self.assertFalse(Path(result["cache_directory"]).is_relative_to(MODULE.parents[1]))

    def test_default_cache_falls_back_when_xdg_is_inside_project(self):
        project_cache = MODULE.parents[1] / ("unused-test-cache-" + self.root.name)
        with (
            mock.patch.dict(os.environ, {"XDG_CACHE_HOME": str(project_cache)}),
            mock.patch.object(Path, "home", return_value=self.root),
        ):
            result = prepare_trace.prepare(
                self.archive, "trace_1", "day:0", "day:1", self.testbed, self.output
            )
        self.assertEqual(
            Path(result["cache_directory"]), self.root / ".cache" / "rnbt" / "trace-data"
        )
        self.assertFalse(project_cache.exists())

    def test_project_cache_is_rejected_without_writing_any_data(self):
        project_cache = MODULE.parents[1] / ("unused-test-cache-" + self.root.name)
        with self.assertRaises(prepare_trace.PreparationError):
            prepare_trace.prepare(
                self.archive,
                "trace_1",
                "day:0",
                "day:1",
                self.testbed,
                self.output,
                cache_dir=project_cache,
            )
        self.assertFalse(project_cache.exists())
        self.assertFalse(self.output.exists())

    def test_cache_and_output_must_not_contain_each_other(self):
        for cache, output in (
            (self.cache, self.cache),
            (self.cache, self.cache / "run"),
            (self.output / "data", self.output),
        ):
            with (
                self.subTest(cache=cache, output=output),
                self.assertRaises(prepare_trace.PreparationError),
            ):
                prepare_trace.prepare(
                    self.archive, "trace_1", "day:0", "day:1", self.testbed, output, cache_dir=cache
                )
            self.assertFalse(self.cache.exists())
            self.assertFalse(self.output.exists())

    def test_cli_accepts_external_cache_directory(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            status = prepare_trace.main(
                [
                    "--archive",
                    str(self.archive),
                    "--trace",
                    "trace_1",
                    "--from",
                    "day:0",
                    "--to",
                    "day:1",
                    "--testbed",
                    str(self.testbed),
                    "--output",
                    str(self.output),
                    "--cache-dir",
                    str(self.cache),
                ]
            )
        self.assertEqual(status, 0)
        self.assertTrue(Path(self.config()["trace"]).is_relative_to(self.cache))
        self.assertTrue(Path(self.config()["workload_profiles"]).is_relative_to(self.cache))

    def test_concurrent_preparations_publish_one_complete_shared_cache(self):
        outputs = (self.output, self.root / "concurrent-run")
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(lambda output: self.prepare(output=output), outputs))
        self.assertEqual(results[0]["trace_directory"], results[1]["trace_directory"])
        self.assertEqual(results[0]["profile_directory"], results[1]["profile_directory"])
        self.assertEqual(self.config(outputs[0])["trace"], self.config(outputs[1])["trace"])
        self.assertEqual(
            self.config(outputs[0])["workload_profiles"],
            self.config(outputs[1])["workload_profiles"],
        )
        trace_directory = Path(results[0]["trace_directory"])
        self.assertEqual((trace_directory / "jobs.jsonl").read_text(), self.files["jobs.jsonl"])
        self.assertEqual(self.profile()["samples"][0]["gpu_util_pct"], 12)
        self.assertEqual(
            {
                path.name
                for path in self.profile_directory().glob("*.json")
                if not path.name.startswith(".")
            },
            {"node-001.json", "node-001.capacity.json"},
        )
        self.assertFalse(
            any(path.is_dir() and ".prepare-" in path.name for path in self.cache.rglob("*"))
        )
        self.assertFalse(
            any(path.is_dir() and ".prepare-" in path.name for path in self.root.iterdir())
        )


if __name__ == "__main__":
    unittest.main()
