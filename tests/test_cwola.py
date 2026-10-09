"""CWoLa label semantics, frozen encoding, batching and saved hyperparameters.

Run from the repository root: python -m unittest discover -s tests -p test_cwola.py
"""
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import torch

from models.cwola import CWoLaMLP
from scripts.cwola_utils import (
    batch_sizes, combine_batches, evaluate, make_loaders, paired_batches, validate_label_sets,
)
from scripts.lejepa_run import LeJEPARun
from scripts.run_train_cwola import resolve_training_config
from scripts.run_train_lejepa_part import make_warmup_cosine_scheduler


def batch(values, ids=None):
    x = torch.as_tensor(values, dtype=torch.float32).reshape(-1, 1, 1)
    result = dict(x_particles=x, padding_mask=torch.zeros(len(x), 1, dtype=torch.bool))
    if ids is not None:
        result['y'] = torch.nn.functional.one_hot(torch.tensor(ids), 3)
    return result


class IdentityScore(torch.nn.Module):
    def forward_logits(self, x):
        return x.reshape(-1)


class CWoLaTests(unittest.TestCase):
    def test_weak_labels_differ_from_truth(self):
        inputs, weak, truth = combine_batches(batch([1.] * 192), batch([2.] * 64))
        self.assertEqual(weak.tolist(), [0.] * 128 + [1.] * 128)
        self.assertEqual(truth.tolist(), [0] * 192 + [1] * 64)
        self.assertEqual(inputs['x_particles'][:, 0, 0].tolist(), [1.] * 192 + [2.] * 64)
        # Dynamic row counts, independent of any configured global batch size.
        _, weak, truth = combine_batches(batch([1.] * 6), batch([2.] * 2))
        self.assertEqual(weak.tolist(), [0.] * 4 + [1.] * 4)
        self.assertEqual(truth.tolist(), [0] * 6 + [1] * 2)

    def test_fraction_boundaries(self):
        self.assertEqual(batch_sizes(256, .5), (192, 64))
        self.assertEqual(batch_sizes(256, 0), (256, 0))
        self.assertEqual(batch_sizes(256, 1), (128, 128))
        self.assertEqual(batch_sizes(256, 0, for_roc=True), (255, 1))
        for fraction in [-.01, 1.01, float('nan'), float('inf')]:
            with self.assertRaises(ValueError):
                batch_sizes(256, fraction)
        with self.assertRaises(ValueError):
            batch_sizes(255, .5)

    def test_disjoint(self):
        with self.assertRaisesRegex(ValueError, 'disjoint'):
            validate_label_sets(dict(background_labels=['QCD', 'Wqq'], signal_labels=['Wqq', 'Hbb']))
        with self.assertRaises(ValueError):
            validate_label_sets(dict(background_labels=['QCD'], signal_labels=[]))
        self.assertEqual(validate_label_sets(dict(background_labels=['QCD', 'Wqq'],
                                                  signal_labels=['Hbb', 'Tbqq'])),
                         (['QCD', 'Wqq'], ['Hbb', 'Tbqq']))

    def test_truth_roc_not_mixture_roc_and_type_alignment(self):
        run = SimpleNamespace(device=torch.device('cpu'), precision='fp32',
                              encode=lambda b: b['x_particles'])
        result = evaluate(IdentityScore(), run,
                          ([batch([-2., -2., -2., -2., -2., -2.])], [batch([2., 2.], [1, 2])]),
                          1, per_signal=True)
        self.assertEqual(result['auc'], 1.)
        self.assertEqual(result['truth'].tolist(), [0] * 6 + [1] * 2)
        self.assertEqual(result['signal_ids'].tolist(), [-1] * 6 + [1, 2])
        self.assertLess(result['scores'][:6].max(), result['scores'][6:].min())

    def test_finite_test_exhaustion_keeps_both_tails(self):
        bg = [batch([1., 1.]), batch([1.])]
        sg = [batch([2.]), batch([2.]), batch([2.])]
        parts = list(paired_batches((bg, sg), 0, require_full_budget=False))
        truths = torch.cat([combine_batches(*pair)[2] for pair in parts])
        self.assertEqual(len(truths), 6)
        self.assertEqual(int(truths.sum()), 3)
        with self.assertRaisesRegex(RuntimeError, 'stream ended'):
            list(paired_batches((bg, sg), 3))

    def test_two_loader_batch_sizes_and_split(self):
        for dataset in ['cms', 'jetclass']:
            run = SimpleNamespace(batch_size=256, summary={}, backgrounds=['QCD', 'Wqq'],
                                  signals=['Hbb', 'Tbqq'], backend=SimpleNamespace(dataset_name=dataset),
                                  loader=Mock())
            make_loaders(run, 'train', .5, steps=5, training=True)
            bg, sg = run.loader.call_args_list
            self.assertEqual(bg.args, ('train', run.backgrounds))
            self.assertEqual(sg.args, ('train', run.signals))
            self.assertEqual(bg.kwargs['batch_size'], 192)
            self.assertEqual(sg.kwargs['batch_size'], 64)
            self.assertTrue(bg.kwargs['infinite'])
            run.loader.reset_mock()
            make_loaders(run, 'test', .5, steps=5, for_roc=True)
            self.assertFalse(run.loader.call_args_list[0].kwargs['infinite'])

    def test_summary_defaults_and_overrides(self):
        defaults = dict(epochs=None, steps_per_epoch=None, val_steps=None, eval_steps=None,
                        learning_rate=None, weight_decay=None, final_lr_ratio=None, batch_size=None,
                        warmup_steps=None)
        summary = dict(epochs=40, steps_per_epoch=2000, val_steps=100, eval_steps=100,
                       batch_size=128, global_batch_size=256, learning_rate=.001,
                       weight_decay=.05, warmup_steps=20000, final_lr_ratio=.001)
        resolved = resolve_training_config(SimpleNamespace(**defaults), summary)
        self.assertEqual(resolved['batch_size'], 256)
        for key in ['epochs', 'learning_rate', 'weight_decay', 'steps_per_epoch', 'warmup_steps']:
            self.assertEqual(resolved[key], summary[key])
        defaults.update(epochs=2, warmup_steps=2, learning_rate=.02)
        resolved = resolve_training_config(SimpleNamespace(**defaults), summary)
        self.assertEqual(resolved['epochs'], 2)
        self.assertEqual(resolved['learning_rate'], .02)
        self.assertEqual(resolved['warmup_steps'], 2)

    def test_frozen_backbone_and_model_output(self):
        class Backbone(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.projection = torch.nn.Linear(1, 128)
            def forward_representation(self, x, padding_mask):
                return self.projection(x[:, 0])
        run = object.__new__(LeJEPARun)
        run.model = Backbone().eval().requires_grad_(False)
        run.device, run.precision = torch.device('cpu'), 'fp32'
        before = {name: p.clone() for name, p in run.model.state_dict().items()}
        model = CWoLaMLP(dropout=0.)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
        x, weak, _ = combine_batches(batch([1.] * 6), batch([2.] * 2))
        z = run.encode(x)
        self.assertFalse(z.requires_grad)
        logits = model.forward_logits(z)
        torch.testing.assert_close(model(z), torch.sigmoid(logits))
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, weak)
        loss.backward()
        optimizer.step()
        self.assertTrue(all(p.grad is None for p in run.model.parameters()))
        for name, p in run.model.state_dict().items():
            torch.testing.assert_close(p, before[name])

    def test_scheduler_endpoints(self):
        model = CWoLaMLP()
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
        scheduler = make_warmup_cosine_scheduler(optimizer, 8, 2, .001)
        self.assertAlmostEqual(optimizer.param_groups[0]['lr'], .0005)
        for _ in range(8):
            optimizer.step()
            scheduler.step()
        self.assertAlmostEqual(optimizer.param_groups[0]['lr'], .000001)


if __name__ == '__main__':
    unittest.main()
