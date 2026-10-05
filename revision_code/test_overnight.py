import unittest
from overnight_queue import dependency_ready,dependency_gpu


class OvernightQueueTests(unittest.TestCase):
    def test_waits_for_active_tde_lane(self):
        self.assertFalse(dependency_ready(dict(status="running",outcomes={"tde_motifs/dev":"complete"})))

    def test_independent_lane_completion_is_enough(self):
        self.assertTrue(dependency_ready(dict(status="running",outcomes={"tde_motifs/test":"complete"})))

    def test_failure_does_not_prevent_independent_gqa_smoke(self):
        self.assertTrue(dependency_ready(dict(status="running",outcomes={"tde_motifs/smoke":"failed"})))
        self.assertEqual(dependency_gpu(dict(status="running",outcomes={"transformer":"reference_gate_failed"})),1)

    def test_finished_queue_releases_dependency(self):
        self.assertTrue(dependency_ready(dict(status="needs_attention",outcomes={})))

    def test_historical_calibration_input_score_semantics(self):
        import torch
        from sgg_core.models.adapters.pysgg_live import PySGGLiveAdapter
        probs=torch.tensor([[.1,.2,.7]])
        logits=torch.tensor([[-3.,1.,4.]])
        self.assertTrue(torch.equal(PySGGLiveAdapter._log_probabilities(probs),probs.log()))
        self.assertTrue(torch.equal(PySGGLiveAdapter._log_probabilities(logits),logits))


if __name__=="__main__":unittest.main()
