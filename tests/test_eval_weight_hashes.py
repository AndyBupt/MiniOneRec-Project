"""Verify launcher records every model shard, supporting FP32 exports over 5GB."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

ROOT = Path(__file__).resolve().parents[1]


class WeightHashTest(unittest.TestCase):
    def test_single_and_sharded_exports(self):
        source = (ROOT / "scripts/eval_valid_single_gpu.sh").read_text()
        code = source.split('<<\'PY\'\n', 1)[1].split('\nPY', 1)[0]
        for sharded in (False, True):
            with self.subTest(sharded=sharded), TemporaryDirectory() as folder:
                root = Path(folder)
                for sub in ("valid", "info"):
                    directory = root / "data/Amazon" / sub
                    directory.mkdir(parents=True)
                    extension = "csv" if sub == "valid" else "txt"
                    (directory / f"Test_5_2016-10-2018-11.{extension}").write_bytes(b"data")
                model = root / "model"
                model.mkdir()
                (model / "tokenizer.json").write_bytes(b"tokenizer")
                if sharded:
                    for name in ("shard1.safetensors", "shard2.safetensors"):
                        (model / name).write_bytes(name.encode())
                    (model / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {
                        "a": "shard1.safetensors", "b": "shard1.safetensors", "c": "shard2.safetensors"}}))
                else:
                    (model / "model.safetensors").write_bytes(b"weights")
                result = subprocess.run([sys.executable, "-c", code, str(model), "Test"],
                                        cwd=root, check=True, capture_output=True, text=True)
                lines = result.stdout.splitlines()
                self.assertEqual(len(lines), 6 if sharded else 4)
                for line in lines:
                    digest, filename = line.split(maxsplit=1)
                    path = Path(filename)
                    if not path.is_absolute():
                        path = root / path
                    self.assertEqual(digest, hashlib.sha256(path.read_bytes()).hexdigest())


if __name__ == "__main__":
    unittest.main()
