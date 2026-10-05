"""Show the bounded matched-control experiments, including terminal failures."""
import argparse
import json
import time
from common import ROOT


def show():
    for family in ['motifs', 'transformer']:
        base = ROOT / 'results/R23_frequency_matched_v2' / family
        done = base / 'formal/summary.json'
        if done.exists():
            row = json.loads(done.read_text())
            print('%s: COMPLETE %d images, %d relations' % (family, row['images'], row['relations']))
            continue
        p = base / 'formal/progress.json'
        if p.exists():
            row = json.loads(p.read_text())
            age = time.time() - p.stat().st_mtime
            eta = row['seconds'] / max(1, row['images']) * (row['total'] - row['images']) / 60
            print('%s: %d/%d; estimated inference remaining %.1f min; last update %.0fs ago' %
                  (family, row['images'], row['total'], eta, age))
        else:
            print(family + ': preparing/smoke; inspect log if this persists')
        log = ROOT / 'logs' / ('R23_'+family+'_frequency_v2.log')
        if log.exists() and 'Traceback' in log.read_text()[-1600:]:
            print('  WARNING: recent traceback; inspect ' + str(log))
    p = ROOT / 'results/R23_semantic_paired/summary.json'
    print('semantic paired: ' + ('COMPLETE' if p.exists() else 'pending'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args()
    while True:
        show()
        if not args.watch:
            break
        time.sleep(30)
        print('\n---')
