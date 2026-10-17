"""Lightweight unit tests; they run without TensorFlow or downloaded wheat datasets."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

MODULE = Path(__file__).resolve().parents[1] / 'src' / 'prepare_generalisation_study.py'
spec = importlib.util.spec_from_file_location('study', MODULE)
study = importlib.util.module_from_spec(spec)
spec.loader.exec_module(study)


class StudyPreparationTests(unittest.TestCase):
    def make_images(self, root, total, code):
        for cls_idx, cls in enumerate(study.CLASSES):
            path = root / cls
            path.mkdir(parents=True, exist_ok=True)
            for i in range(total):
                # Distinct binary content (not three repeated uniform images).
                im = Image.new('RGB', (20, 20))
                pix = im.load()
                for x in range(20):
                    for y in range(20):
                        pix[x, y] = ((code*37+cls_idx*57+i*11+x*5)%255,
                                     (code*11+i*23+y*9)%255, (i*37+x+y*2)%255)
                im.save(path / f'{cls}_{i}.png')

    def test_balanced_provenance_and_exact_leakage_exclusion(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            self.make_images(d/'primary', 8, 1)
            self.make_images(d/'other', 7, 2)
            self.make_images(d/'internal', 3, 3)
            self.make_images(d/'external', 3, 4)
            # Deliberately contaminated sample from the held-out set.
            leak = (d/'external'/'Healthy'/'Healthy_0.png').read_bytes()
            (d/'other'/'Healthy'/'copied-leak.png').write_bytes(leak)
            study.main(['--primary-train', str(d/'primary'), '--additional-source', f"iari={d/'other'}",
                '--internal-val', str(d/'internal'), '--external-test', str(d/'external'),
                '--per-class', '4', '--mix', '0', '0.5', '--seed', '42', '43',
                '--out-dir', str(d/'staged'), '--phash-threshold', '-1'])
            manifest = json.loads((d/'staged'/'study_manifest.json').read_text())
            assert len(manifest['conditions']) == 4
            assert any(x['reason']=='exact_holdout_overlap' for x in manifest['filtered_candidates'])
            for condition in manifest['conditions']:
                rows = json.loads((d/'staged'/condition['id']/'selected_images.json').read_text())
                assert len(rows) == 12
                assert all(sum(r['class']==cls for r in rows)==4 for cls in study.CLASSES)
                assert all(r['sha256'] != study.fingerprint(d/'external'/'Healthy'/'Healthy_0.png', False)['sha256'] for r in rows)
                if condition['mix'] == .5:
                    assert sum(r['source']=='iari' for r in rows) == 6

    def test_cross_source_exact_duplicates_filtered_with_phash_enabled(self):
        r = lambda name, sha, ph: {'path': Path(name), 'sha256':sha, 'phash':ph}
        empty = lambda: {k: [] for k in study.CLASSES}
        primary = empty()
        extra = empty()
        primary['Healthy']=[r('a.png','same',0b101)]
        extra['Healthy']=[r('b.png','same',0b101)]
        kept, dropped = study.check_and_filter_sources(
            {'primary':primary,'other':extra},empty(),empty(),5)
        assert len(kept['primary']['Healthy']) == 1
        assert len(kept['other']['Healthy']) == 0
        assert len(dropped) == 1 and dropped[0]['reason'] == 'exact_training_duplicate'

    def test_perceptual_filter(self):
        r = lambda name, sha, ph: {'path': Path(name), 'sha256':sha, 'phash':ph}
        hold = {k: [] for k in study.CLASSES}
        hold['Healthy']=[r('holdout.png','held',0b0101)]
        sources={'primary': {k: [] for k in study.CLASSES}}
        sources['primary']['Healthy']=[r('new.png','new',0b0100)]
        kept, dropped = study.check_and_filter_sources(sources, hold, {k: [] for k in study.CLASSES}, 1)
        assert len(dropped)==1 and dropped[0]['reason']=='perceptual_holdout_overlap'
        assert not kept['primary']['Healthy']

if __name__ == '__main__':
    unittest.main()
