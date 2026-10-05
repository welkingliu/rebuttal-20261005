import copy
from contextlib import redirect_stdout
import io
import unittest

from show_status import show, unfinished_snapshot


class UnfinishedDashboardTests(unittest.TestCase):
    def snapshot(self):
        return dict(
            time="2026-10-03 15:00:00", gpu=[],
            active=[dict(name="R16_gpu0", status="running", pid=1),
                    dict(name="VK_queue", status="failed", reason="runtime failure"),
                    dict(name="R12_transformer_eligibility", status="blocked"),
                    dict(name="finished_job", status="complete")],
            queue=dict(status="complete"),
            identity_queue=dict(status="needs_attention"),
            external_queue=dict(status="complete"),
            followups=dict(R15=dict(status="complete"),
                           VI=dict(status="complete", accepted=False),
                           VK=dict(status="failed")),
            recent={"R14 corrected SGDet test": "PASSED: registered reference tolerance",
                    "R15 Transformer identity": "FINISHED: results available"},
        )

    def test_filters_history_without_mutating_snapshot(self):
        original = self.snapshot()
        saved = copy.deepcopy(original)
        result = unfinished_snapshot(original)
        self.assertEqual([r["name"] for r in result["active"]], ["R16_gpu0", "VK_queue"])
        self.assertEqual(set(result["followups"]), {"VK"})
        for key in ("recent", "queue", "identity_queue", "external_queue"):
            self.assertEqual(result[key], {})
        self.assertEqual(original, saved)

    def test_does_not_hide_unresolved_historical_blocker(self):
        data = self.snapshot()
        data["recent"].pop("R14 corrected SGDet test")
        result = unfinished_snapshot(data)
        self.assertIn("R12_transformer_eligibility", [r["name"] for r in result["active"]])
        self.assertEqual(result["identity_queue"]["status"], "needs_attention")

    def test_retains_waiting_pending_failed_and_stale(self):
        data = self.snapshot()
        states = ["waiting_gpu", "waiting_dependency", "pending", "queued",
                  "failed", "blocked", "needs_attention", "STALE: queue process missing"]
        data["active"] = [dict(name=s, status=s) for s in states]
        self.assertEqual([r["status"] for r in unfinished_snapshot(data)["active"]], states)

    def test_compact_text_has_no_finished_results_or_duplicate_queue(self):
        stream = io.StringIO()
        with redirect_stdout(stream):
            show(unfinished_snapshot(self.snapshot()))
        output = stream.getvalue()
        for text in ("RECENT RESULTS", "FOLLOW-UP VK", "R9 TRAINING", "NEXT QUEUE",
                     "R12_transformer_eligibility", "finished_job", "Completed stages"):
            self.assertNotIn(text, output)
        self.assertIn("VK_queue | failed", output)
        self.assertIn("R16_gpu0 | running", output)

    def test_history_view_still_available(self):
        stream = io.StringIO()
        with redirect_stdout(stream):
            show(self.snapshot(), include_history=True)
        self.assertIn("RECENT RESULTS", stream.getvalue())

    def test_failed_stage_has_no_misleading_zero_eta(self):
        data = unfinished_snapshot(self.snapshot())
        data["active"] = [dict(name="VK_queue", status="failed",
                               progress=dict(images=0, total=3, eta_seconds=0))]
        stream = io.StringIO()
        with redirect_stdout(stream):
            show(data)
        self.assertIn("0/3", stream.getvalue())
        self.assertNotIn("remaining 0m", stream.getvalue())


if __name__ == "__main__":
    unittest.main()
