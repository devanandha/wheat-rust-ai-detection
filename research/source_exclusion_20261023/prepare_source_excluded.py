"""Reconstructed from the PowerShell procedure executed on 22 October 2026.

Not the original executed file. Uses the archived preparation implementation;
never trains a model. Supply existing, locally valid dataset/manifest paths.
"""
import argparse
from collections import defaultdict
from datetime import datetime
import json
from pathlib import Path
import random
from src import prepare_generalisation_study as prep


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-manifest', type=Path, required=True)
    parser.add_argument('--staging-root', type=Path, required=True)
    parser.add_argument('--iari-root', type=Path, required=True,
                        help='WheatLeafRust root containing val and test')
    parser.add_argument('--out-dir', type=Path, required=True)
    args = parser.parse_args()
    previous = json.loads(args.baseline_manifest.read_text(encoding='utf-8-sig'))
    if args.out_dir.exists():
        raise FileExistsError(args.out_dir)
    protected = defaultdict(list)
    for split in ('val', 'test'):
        for folder, cls in (('control', 'Healthy'), ('diseased', 'Brown_rust')):
            root = args.iari_root / split / folder
            if not root.is_dir():
                raise ValueError(f'Missing protected folder: {root}')
            images = sorted(p for p in root.iterdir()
                            if p.is_file() and p.suffix.lower() in prep.SUFFIXES)
            if not images:
                raise ValueError(f'No images in {root}')
            protected[cls].extend(prep.fingerprint(p, True) for p in images)
    ethiopia = prep.inventory(args.staging_root / 'holdouts/ethiopia', True, False)
    for cls in prep.CLASSES:
        protected[cls].extend(ethiopia[cls])
    original_filter = prep.check_and_filter_sources
    original_select = prep.select_cohort

    def screen(sources, internal, external, threshold):
        original_filter(sources, internal, external, threshold)
        combined = defaultdict(list)
        seen = set()
        for pool in (internal, external, protected):
            for cls in prep.CLASSES:
                for row in pool[cls]:
                    if row['sha256'] not in seen:
                        seen.add(row['sha256'])
                        combined[cls].append(row)
        return original_filter(sources, combined, defaultdict(list), threshold)

    def select(kept, size, mix, seed):
        rng = random.Random(seed)
        extra = {name: pool for name, pool in kept.items() if name != 'primary'}
        selected = []
        for cls in prep.CLASSES:
            available = sum(len(pool[cls]) for pool in extra.values())
            needed = round(size * mix) if available else 0
            if available < needed:
                raise ValueError(f'Insufficient screened extra images for {cls}: {available}')
            selected.extend(('primary', cls, row) for row in
                            rng.sample(kept['primary'][cls], size-needed))
            selected.extend((name, cls, row) for name, row in
                            prep._sample_extra(extra, cls, needed, rng))
        return selected

    prep.check_and_filter_sources = screen
    prep.select_cohort = select
    try:
        for added, excluded in (('iari', 'ethiopia'), ('ethiopia', 'iari')):
            destination = args.out_dir / f'add_{added}_exclude_{excluded}'
            prep.main([
                '--primary-train', previous['train_primary'],
                '--internal-val', previous['internal_validation'],
                '--external-test', previous['external_heldout_test'],
                '--additional-source', f'{added}={args.staging_root / "sources" / added}',
                '--per-class', '150', '--mix', '0.25', '--seed', '42', '43', '44',
                '--out-dir', str(destination), '--copy-mode', 'copy', '--phash-threshold', '5',
            ])
            path = destination / 'study_manifest.json'
            manifest = json.loads(path.read_text(encoding='utf-8'))
            manifest['protocol'] = 'Exploratory source-excluded, class-balanced comparison'
            manifest['excluded_training_source'] = excluded
            manifest['mix_policy'] = ('38 additional and 112 primary per supported class; '
                                      '150 primary for classes absent from the additional source.')
            manifest['additional_protected_sets'] = [str(args.iari_root / 'val'),
                str(args.iari_root / 'test'), str(args.staging_root / 'holdouts/ethiopia')]
            path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
            for condition in manifest['conditions']:
                print(added, condition['id'], condition['counts_by_source_class'])
    finally:
        prep.check_and_filter_sources = original_filter
        prep.select_cohort = original_select
    print('PREPARATION COMPLETE:', args.out_dir)


if __name__ == '__main__':
    main()
