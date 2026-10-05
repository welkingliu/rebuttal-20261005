"""Reanalyse all registered semantic corruption contrasts, without new inference."""
import json
from pathlib import Path
import numpy as np
from common import ROOT, ensure_storage, atomic_json, sha256
from evidence_completion import paired_ratio


def main():
    ensure_storage()
    result = {}
    for family, relative in [('motifs', 'R20_plain_motifs_native_bridge/identity'),
                             ('transformer', 'R15_transformer_identity/transformer/test')]:
        source = ROOT / 'results' / relative
        protocol = json.loads((source / 'protocol.json').read_text())
        digest = sha256(source / 'protocol.json')
        conditions = [(seed, strength) for seed in (17, 23, 31) for strength in (25, 50, 100)]
        rows = {str(c): [] for c in conditions}
        for iid in protocol['image_ids']:
            metadata = json.loads((source / 'images' / (iid + '.json')).read_text())
            if metadata['protocol_sha256'] != digest or not metadata['invariance_passed']:
                raise RuntimeError('Parent protocol/invariance mismatch')
            with np.load(source / 'images' / (iid + '.npz')) as z:
                truth = z['relation_gt']
                if not len(truth):
                    continue
                for seed, strength in conditions:
                    suffix = ('embedding_' if family == 'transformer' else '') + 's%d_p%d' % (seed, strength)
                    a = int((z['prediction_confusable_' + suffix] == truth).sum())
                    b = int((z['prediction_matched_random_' + suffix] == truth).sum())
                    rows[str((seed, strength))].append(dict(image_id=iid, confusable=a, random=b, n=len(truth)))
        estimates = {}
        for key, items in rows.items():
            estimates[key] = paired_ratio([x['confusable'] for x in items], [x['random'] for x in items],
                                         [x['n'] for x in items], draws=10000, seed=17)
        out = ROOT / 'results/R23_semantic_paired' / family
        atomic_json(out / 'counts.json', rows)
        result[family] = dict(status='complete', contrasts=estimates, parent_protocol_sha256=digest,
                              source_sha256=sha256(Path(__file__)),
                              scope='Post-hoc; no training-seed replication or multiplicity correction')
        atomic_json(out / 'summary.json', result[family])
        print('[COMPLETE] semantic paired ' + family, flush=True)
    atomic_json(ROOT / 'results/R23_semantic_paired/summary.json', result)


if __name__ == '__main__':
    main()
