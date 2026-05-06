#!/usr/bin/env python3

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest


if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
    LOAD_BINARY = str(Path(sys.argv.pop(1)).resolve())
else:
    LOAD_BINARY = os.environ.get("RNBT_LOAD_BINARY", "")


class LoadCliTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(LOAD_BINARY, "Pass the rnbt-load executable path as the first argument")
        self.directory = tempfile.TemporaryDirectory(prefix="rnbt-load-cli-")
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.state_path = self.root / "state.json"

    def invoke(self, *arguments):
        return subprocess.run([LOAD_BINARY, *arguments], capture_output=True, text=True, timeout=15)

    def write_state(self, status="running", error="", pid=99_999_999):
        state = {
            "schema_version": 1,
            "program": "rnbt-load",
            "status": status,
            "error": error,
            "pid": pid,
            "process_start_ticks": "identity-must-not-match-a-real-process",
            "executable": LOAD_BINARY,
            "samples_written": 42,
            "node": "node-001",
            "profile": str(self.root / "profile.json"),
            "log": str(self.root / "samples.jsonl"),
            "start_wall_ns": time.time_ns(),
            "gpus": 1,
        }
        self.state_path.write_text(json.dumps(state), encoding="utf-8")
        return state

    def read_state(self):
        return json.loads(self.state_path.read_text(encoding="utf-8"))

    def test_stop_dead_actor_preserves_failure_and_sample_count(self):
        for status in ("running", "starting"):
            with self.subTest(status=status):
                self.write_state(status=status)
                stopped = self.invoke("stop", "--state", str(self.state_path))
                self.assertEqual(stopped.returncode, 0, stopped.stderr)
                current = self.read_state()
                self.assertEqual(current["status"], "failed")
                self.assertTrue(current["error"])
                self.assertEqual(current["samples_written"], 42)
                reported = self.invoke("status", "--state", str(self.state_path))
                self.assertNotEqual(reported.returncode, 0)
                self.assertEqual(json.loads(reported.stdout)["status"], "failed")

    def test_stop_does_not_signal_a_process_with_mismatched_identity(self):
        process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        try:
            self.write_state(pid=process.pid)
            stopped = self.invoke("stop", "--state", str(self.state_path))
            self.assertEqual(stopped.returncode, 0, stopped.stderr)
            self.assertIsNone(process.poll(), "An unrelated process must remain alive")
            self.assertEqual(self.read_state()["status"], "failed")
        finally:
            process.terminate()
            process.wait(timeout=5)

    def test_stop_preserves_existing_failure(self):
        original = self.write_state(status="failed", error="original CUDA allocation failure")
        for _ in range(2):
            stopped = self.invoke("stop", "--state", str(self.state_path))
            self.assertEqual(stopped.returncode, 0, stopped.stderr)
            current = self.read_state()
            self.assertEqual(current["status"], "failed")
            self.assertEqual(current["error"], original["error"])
            self.assertEqual(current["samples_written"], 42)

    def test_stop_preserves_completed_state(self):
        self.write_state(status="completed")
        stopped = self.invoke("stop", "--state", str(self.state_path))
        self.assertEqual(stopped.returncode, 0, stopped.stderr)
        self.assertEqual(self.read_state()["status"], "completed")
        self.assertEqual(self.read_state()["samples_written"], 42)
        reported = self.invoke("status", "--state", str(self.state_path))
        self.assertEqual(reported.returncode, 0, reported.stderr)

    def test_stop_with_missing_state_is_idempotent(self):
        for _ in range(2):
            stopped = self.invoke("stop", "--state", str(self.state_path))
            self.assertEqual(stopped.returncode, 0, stopped.stderr)
        self.assertFalse(self.state_path.exists())

    def test_rejects_invalid_arguments_and_state(self):
        for arguments in (
            [],
            ["unknown"],
            ["start"],
            ["stop"],
            ["status"],
            ["stop", "--state", str(self.state_path), "--unexpected", "value"],
            ["start", "--gpus", "0"],
        ):
            with self.subTest(arguments=arguments):
                self.assertNotEqual(self.invoke(*arguments).returncode, 0)
        self.state_path.write_text('{"status":"running"}', encoding="utf-8")
        self.assertNotEqual(self.invoke("stop", "--state", str(self.state_path)).returncode, 0)
        self.assertEqual(self.read_state(), {"status": "running"})


if __name__ == "__main__":
    unittest.main()
