import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import awkward as ak
import numpy as np
import torch
import uproot
from torch.utils.data import DataLoader
from datasets import cms_streaming as cms
from scripts import evaluate_label_real_anomalies as ev


def write_root(path, n, offset=0, invalid_first=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for key in cms._REQUIRED_PARTICLE_BRANCHES:
        if key.startswith('jet_'):
            arrays[key] = np.full(n, 500. if key == 'jet_pt' else 0., dtype=np.float32)
        else:
            values = np.ones((n, 2), dtype=np.float32)
            if any(key.endswith(suffix) for suffix in ('eta', 'phi', 'etarel', 'phirel')):
                values[:] = [.1, -.2]
            if key.endswith('_pdg'): values[:] = 211
            if key.endswith('_pdgID'): values[:] = 130
            if key.endswith('_isGamma'): values[:] = 0
            if key.endswith(('_px', '_pt', '_e')):
                values[:] = np.arange(n)[:, None] + offset + 2
            arrays[key] = ak.Array(values.tolist())
    if invalid_first:
        arrays['jet_pt'][0] = -1
    with uproot.recreate(path) as handle:
        handle['tree'] = arrays


class MixedEvaluationTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.qcd = self.root/'qcd/qcd_0.root'
        self.real = self.root/'real/data_0.root'
        self.test = self.root/'real/data_test.root'
        write_root(self.qcd, 3, invalid_first=True)
        write_root(self.real, 5, offset=10)
        write_root(self.test, 4, offset=20)
        self.files = {'label_QCD': [str(self.qcd)], 'label_Real': [str(self.real)]}

    def dataset(self, **kwargs):
        return cms.CMSIterableDataset(self.files, list(self.files), cms.CMS_LABELS,
                cms.PARTICLE_FEATURES, 8, infinite=False,
                stop_on_first_exhaustion=True, **kwargs)

    def test_finite_balanced_and_partial_batch(self):
        batches = list(DataLoader(self.dataset(max_events=30), batch_size=3,
                       num_workers=0, collate_fn=cms.collate_cms_tensors))
        self.assertEqual([len(b['y']) for b in batches], [3, 1])
        ids = torch.cat([b['y'].argmax(1) for b in batches])
        self.assertEqual(ids.tolist(), [0, 5, 0, 5])
        values = torch.cat([b['x_particles'][:, 0, 0] for b in batches])
        self.assertEqual(len(values.unique()), 4)
        self.assertEqual(len(list(self.dataset(max_events=3))), 2)
        with self.assertRaisesRegex(ValueError, 'one rank'):
            list(self.dataset(world_size=2))
        from types import SimpleNamespace
        with patch.object(cms, 'get_worker_info', return_value=SimpleNamespace(id=0, num_workers=2)):
            with self.assertRaisesRegex(ValueError, 'one rank'):
                list(self.dataset(max_events=30))
        regular = cms.CMSIterableDataset(self.files, list(self.files), cms.CMS_LABELS,
                cms.PARTICLE_FEATURES, 8, infinite=False, shuffle_files=True)
        self.assertEqual(len(list(regular)), 7)
        # Empty slices must not refill or fail; an entirely empty class ends the pool.
        write_root(self.qcd, 1, invalid_first=True)
        self.assertEqual(list(self.dataset(max_events=30)), [])

    def test_split_inheritance(self):
        files = {}
        for label, directory in [('label_QCD', 'qcd'), ('label_Real', 'real')]:
            paths=[]
            for i in range(20):
                p=self.root/directory/f'fixture_{i}.root'; p.write_bytes(b'x'); paths.append(str(p))
            files[label]=paths
        original = cms.split_cms_files({'label_QCD':files['label_QCD']}, seed=13)
        manifest=cms.cms_split_manifest(original)
        source=self.root/'pretrain'; source.mkdir()
        (source/'cms_split_manifest.json').write_text(json.dumps(manifest))
        summary=dict(dataset='cms', dataset_root=str(self.root), cms_val_fraction=.1,
                     cms_test_fraction=.1, cms_split_manifest_sha256=manifest['sha256'])
        (source/'summary.json').write_text(json.dumps(summary))
        splits, mc, real, info=cms.prepare_cms_splits(files, dataset_root=self.root,
               val_fraction=.1, test_fraction=.1, seed=99, checkpoint=source/'best_model.pth')
        for split in original:
            self.assertEqual(splits[split]['label_QCD'], original[split]['label_QCD'])
        self.assertEqual([len(splits[x]['label_Real']) for x in ('train','val','test')], [10,1,9])
        self.assertIsNotNone(info['source'])
        fresh, _, _, info=cms.prepare_cms_splits(files, dataset_root=self.root,
               val_fraction=.2, test_fraction=.1, checkpoint=source/'best_model.pth')
        self.assertIsNone(info['source'])
        self.assertEqual(len(fresh['val']['label_QCD']),4)
        with self.assertRaises(ValueError):
            cms.prepare_cms_splits(files, dataset_root=self.root, val_fraction=.1,
                test_fraction=.1, real_val_fraction=.7, real_test_fraction=.5)
        (source/'cms_split_manifest.json').unlink()
        with self.assertRaises(FileNotFoundError):
            cms.prepare_cms_splits(files, dataset_root=self.root, val_fraction=.1,
                test_fraction=.1, checkpoint=source/'best_model.pth')

    def test_knn_and_world_guard(self):
        reference=torch.tensor([[0.,0.],[2.,0.],[4.,0.],[7.,0.]])
        query=torch.tensor([[1.,0.],[6.,0.]])
        for reduction in ('mean','kth'):
            best=torch.cdist(query,reference).topk(3,largest=False).values
            expected=best.mean(1) if reduction=='mean' else best.max(1).values
            actual=ev.knn_scores(query,reference,k=3,reduction=reduction,reference_batch_size=2)
            np.testing.assert_allclose(actual,expected.numpy())
        with self.assertRaises(ValueError):
            ev.knn_scores(query,reference,k=5,reduction='mean',reference_batch_size=2)
        with patch.dict(os.environ, WORLD_SIZE='2'):
            with self.assertRaisesRegex(RuntimeError,'world size'):
                ev.validate_world_size()

    def test_full_cli_one_spawn_worker(self):
        run=self.root/'run'; run.mkdir()
        splits={'train':self.files, 'val':{'label_QCD':[], 'label_Real':[]},
                'test':{'label_QCD':[], 'label_Real':[str(self.test)]}}
        manifest=cms.cms_split_manifest(splits)
        (run/'cms_split_manifest.json').write_text(json.dumps(manifest))
        summary=dict(dataset='cms', dataset_root=str(self.root),
            background_labels=list(self.files), signal_labels=[], dataset_label_axis=cms.CMS_LABELS,
            particle_features=cms.PARTICLE_FEATURES,
            batch_standardized_particle_features=cms.BATCH_NORMALIZED_FEATURES,
            cms_split_manifest_sha256=manifest['sha256'], embed_dim=16, representation_dim=8,
            num_heads=2,num_layers=1,num_class_layers=1,max_num_particles=8,min_nodes=4,
            precision='fp32',batch_size=3,num_workers=4, use_pairwise_bias=False)
        (run/'summary.json').write_text(json.dumps(summary))
        backend=ev.RealTestBackend(summary=summary,run_dir=run,dataset_root_override=None,
                manifest_override=None,max_num_particles_override=None)
        model=ev.build_model(summary,backend,torch.device('cpu'))
        torch.save(model.state_dict(),run/'best_model.pth')
        result=subprocess.run([sys.executable,'scripts/evaluate_label_real_anomalies.py',str(run),
            '--device','cpu','--reference-steps','10','--knn-k','2','--top-fraction','.5',
            '--num-visualize','2'],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stdout+'\n'+result.stderr)
        self.assertIn('to 1', result.stderr)
        self.assertIn('m=2',result.stderr)
        output=run/'real_data_evaluation'
        stats=json.loads((output/'real_test_evaluation.json').read_text())
        self.assertEqual(stats['reference_counts'],{'label_QCD':2,'label_Real':2})
        self.assertEqual(stats['num_workers'],1)
        self.assertEqual(stats['total_test_events'],4)
        self.assertEqual(stats['top_count'],2)
        scores=np.load(output/'real_test_knn_scores.npy')
        selected=np.load(output/'real_test_top_50pct_event_scores.npy')
        np.testing.assert_allclose(selected,np.sort(scores)[::-1][:2])
        self.assertTrue(np.isfinite(scores).all())
        self.assertTrue((output/'real_test_top_events_random_visualization.png').is_file())
        bottom=json.loads((output/'real_test_bottom_evaluation.json').read_text())
        self.assertEqual(bottom['bottom_fraction_requested'],.5)
        self.assertEqual(bottom['bottom_count'],2)
        bottom_scores=np.load(output/'real_test_bottom_50pct_event_scores.npy')
        np.testing.assert_allclose(bottom_scores,np.sort(scores)[:2])
        self.assertTrue((output/'real_test_bottom_events_random_visualization.png').is_file())
        self.assertTrue((output/'real_test_bottom_score_distribution.png').is_file())
        indices=bottom['random_visualized_event_array_indices']
        self.assertEqual(indices, sorted(indices))



if __name__=='__main__':
    unittest.main()
