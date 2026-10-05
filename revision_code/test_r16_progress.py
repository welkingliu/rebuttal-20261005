import json
import os
from pathlib import Path
import tempfile
import unittest

from show_status import r16_progress_path


class ProgressTests(unittest.TestCase):
    def test_validation_then_training_then_test(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            main = root / "progress.json"
            val = root / "validation_003000/progress.json"
            test = root / "test/progress.json"
            for i, path in enumerate([main, val]):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps({"stage":"training" if path == main else "evaluation"}))
                os.utime(str(path), (100+i,100+i))
            self.assertEqual(r16_progress_path(main), val)
            os.utime(str(main), (102,102))
            self.assertEqual(r16_progress_path(main), main)
            test.parent.mkdir()
            test.write_text("{}")
            os.utime(str(test), (103,103))
            self.assertEqual(r16_progress_path(main), test)


if __name__ == "__main__": unittest.main()
