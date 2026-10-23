"""Synthetic ROOT integration checks for fQCD and its plotting entry point."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import awkward as ak
import numpy as np
import torch
import uproot

from datasets import jetclass_streaming as jc


class FakeQCDTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.shard = self.root / "ZJetsToNuNu_000.root"
        # Interior, outside below/above/right, boundary right/bottom, radius cut.
        eta = [0.1, 0.1, 0.1, 0.5, 0.4, 0.0, 0.9]
        phi = [-0.1, -0.5, 0.1, -0.1, -0.1, -0.4, 0.0]
        branches = {name: ak.Array([[1.] * 7] * 4) for name in jc.RAW_PARTICLE_FEATURES}
        branches['part_deta'] = ak.Array([eta] * 4)
        branches['part_dphi'] = ak.Array([phi] * 4)
        branches['part_d0val'] = ak.Array([[-1.] * 7] * 4)
        branches.update({name: np.full(4, value, dtype=np.float32) for name, value in
                         zip(jc.REQUIRED_JET_FEATURES, [100., 0., 0., 200.])})
        branches.update({name: np.full(4, int(name == 'label_QCD'), dtype=np.int32)
                         for name in jc.JETCLASS_ROOT_LABELS})
        with uproot.recreate(self.shard) as f:
            f['tree'] = branches

    def load(self, label, features=None):
        return jc.load_and_preprocess_jetclass_file(
            str(self.shard), features or jc.PARTICLE_FEATURES, 8, label_name=label)

    def test_mask_labels_and_collation(self):
        with patch.object(jc, 'read_file', wraps=jc.read_file) as read:
            fake, y = self.load('label_fQCD')
            self.assertNotIn('label_fQCD', read.call_args.kwargs['labels'])
        real, real_y = self.load('label_QCD')
        expected = np.array([True, False, False, False, True, True, True, True])
        np.testing.assert_array_equal(np.all(fake == 0, axis=-1), np.tile(expected, (4, 1)))
        np.testing.assert_array_equal(fake[:, ~expected], real[:, ~expected])
        self.assertTrue(np.all(y[:, -1] == 1))
        self.assertTrue(np.all(y.sum(axis=1) == 1))
        self.assertTrue(np.all(real_y[:, 0] == 1))
        self.assertTrue(np.all(real_y[:, -1] == 0))
        batch = jc.collate_jetclass_tensors([(torch.from_numpy(fake[0]), torch.from_numpy(y[0]))])
        np.testing.assert_array_equal(batch['padding_mask'][0].numpy(), expected)
        # Geometry must work even when coordinates are not exported to the model.
        subset, _ = self.load('label_fQCD', ['part_px', 'part_energy'])
        np.testing.assert_array_equal(np.all(subset == 0, axis=-1), np.tile(expected, (4, 1)))

    def test_balanced_stream_in_both_modes(self):
        for shuffled in (False, True):
            ds = jc.JetClassIterableDataset(str(self.root), ['label_QCD', 'label_fQCD'],
                    jc.PARTICLE_FEATURES, 8, max_events=8, shuffle_files=shuffled, seed=7)
            events = list(ds)
            ids = [int(y.argmax()) for _, y in events]
            self.assertEqual(len(events), 8)
            for start in range(0, 8, 2):
                self.assertEqual(set(ids[start:start+2]), {0, 10})
            for x, y in events:
                self.assertEqual(bool(x[0].eq(0).all()), int(y.argmax()) == 10)

    def test_legacy_axis_and_plot(self):
        from scripts.diagnose_lejepa_latents import DiagnosticDatasetBackend
        from visualize.plot_random_jet_events import plot_random_jet_events
        from scripts.evaluate_label_real_anomalies import plot_random_jet_events as imported
        self.assertIs(imported, plot_random_jet_events)
        (self.root / 'val_5M').mkdir()
        backend = DiagnosticDatasetBackend.from_summary(summary={
            'dataset': 'jetclass', 'dataset_root': str(self.root),
            'particle_features': jc.PARTICLE_FEATURES,
            'batch_standardized_particle_features': jc.BATCH_NORMALIZED_FEATURES,
            'dataset_label_axis': jc.JETCLASS_ROOT_LABELS,
        }, run_dir=self.root, dataset_override=None, dataset_root_override=None,
           cms_manifest_override=None, max_num_particles_override=None)
        self.assertEqual(backend.label_axis, jc.JETCLASS_LABELS)
        backend.validate_requested_labels(['label_fQCD'])
        x, _ = self.load('label_fQCD')
        output = self.root / 'fake.png'
        selected = plot_random_jet_events(events=x, feature_names=jc.PARTICLE_FEATURES,
                output_path=output, num_events=4, seed=4, nrows=2, ncols=2,
                event_labels=['label_fQCD'] * 4)
        self.assertEqual(len(selected), 4)
        self.assertGreater(output.stat().st_size, 1000)
        # The plotting CLI requires only data, with no run metadata/checkpoint.
        import sys
        import shutil
        from visualize.plot_random_jet_events import main
        shutil.copy2(self.shard, self.root / 'val_5M' / self.shard.name)
        with patch.object(sys, 'argv', ['plot_random_jet_events.py',
                '--dataset-root', str(self.root), '--dataset', 'jetclass',
                '--labels', 'label_fQCD', '--rows', '2', '--cols', '2',
                '--sample-events', '4', '--output', str(self.root / 'cli.png')]):
            main()
        self.assertGreater((self.root / 'cli.png').stat().st_size, 1000)
        self.assertFalse((self.root / 'cli.latents.npy').exists())
        # Original file-based CMS caller remains supported.
        features = [dict(d0_sig='Cpfcan_dxysig', dz_sig='Cpfcan_dzsig').get(n, n)
                    for n in jc.PARTICLE_FEATURES]
        np.save(self.root / 'cms.npy', np.abs(x))
        plot_random_jet_events(events_path=self.root / 'cms.npy', feature_names=features,
                output_path=self.root / 'cms.png', num_events=2, seed=4)


if __name__ == '__main__':
    unittest.main()
