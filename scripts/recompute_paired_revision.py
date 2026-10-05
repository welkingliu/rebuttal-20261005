"""Recompute the published paired conditional contrast from image counts."""
import json
from pathlib import Path
import numpy as np

root=Path(__file__).resolve().parents[1]
expected=json.loads((root/'evidence/paired_channels.json').read_text())
for name, reference in expected.items():
    data=json.loads((root/'evidence'/(name+'_counts.json')).read_text())
    rows=np.asarray(data['counts'],dtype=float)
    rows=rows[rows[:,1]>0]
    rng=np.random.default_rng(17)
    draws=rng.integers(0,len(rows),size=(10000,len(rows)))
    totals=rows[draws].sum(1)
    ci=np.quantile(totals[:,0]/totals[:,1],[.025,.975])
    value=rows[:,0].sum()/rows[:,1].sum()
    assert np.isclose(value,reference['statistic']['value'],atol=1e-12,rtol=0)
    assert np.allclose(ci,reference['statistic']['bootstrap_95ci'],atol=1e-12,rtol=0)
    print(name,'delta_pp=',value*100,'95ci_pp=',(ci*100).tolist())
