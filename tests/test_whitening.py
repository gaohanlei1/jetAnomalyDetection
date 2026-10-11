"""Background-only fitting, streaming covariance, persistence and inference parity."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np
import torch

from models.latent_whitening import BackgroundWhitening
from scripts.cwola_utils import encode_cwola, evaluate, fit_background_whitening, restore_whitening


def raw_batch(values):
    x = torch.as_tensor(values, dtype=torch.float32).reshape(-1, 1, 1)
    return dict(x_particles=x, padding_mask=torch.zeros(len(x), 1, dtype=torch.bool))


class WhiteningTests(unittest.TestCase):
    def test_streaming_covariance_whitens_and_preserves_dimension(self):
        generator = torch.Generator().manual_seed(32)
        z = torch.randn(2000, 4, generator=generator, dtype=torch.float64)
        a = torch.tensor([[3., 1., 0., 0.], [0., 2., .5, 0.], [0., 0., 1., .5], [0., 0., 0., 2.]], dtype=torch.float64)
        x = z @ a + torch.tensor([20., -4., 3., 10.])
        transform, metadata = BackgroundWhitening.fit(x.split(137), eps=1e-8)
        entire, _ = BackgroundWhitening.fit([x], eps=1e-8)
        torch.testing.assert_close(transform.matrix, entire.matrix)
        y = transform(x).double()
        self.assertEqual(y.shape, x.shape)
        torch.testing.assert_close(y.mean(0), torch.zeros(4, dtype=torch.float64), atol=2e-6, rtol=0)
        torch.testing.assert_close(torch.cov(y.T), torch.eye(4, dtype=torch.float64), atol=2e-6, rtol=0)
        self.assertEqual(metadata['num_events'], 2000)
        self.assertEqual(list(transform.parameters()), [])

    def test_singular_covariance_and_invalid_inputs(self):
        transform, metadata = BackgroundWhitening.fit([torch.ones(20, 4)])
        self.assertTrue(torch.isfinite(transform(torch.zeros(3, 4))).all())
        self.assertEqual(metadata['clipped_eigenvalues'], 4)
        for batches in ([], [torch.ones(1, 4)], [torch.full((2, 4), float('nan'))]):
            with self.assertRaises(ValueError):
                BackgroundWhitening.fit(batches)

    def test_persistence_and_required_artifact(self):
        transform = BackgroundWhitening(torch.tensor([2., 3.]), torch.eye(2) * 2)
        with TemporaryDirectory() as directory:
            directory = Path(directory)
            transform.save(directory / 'latent_whitening.pt', {'num_events': 20})
            run = SimpleNamespace(model=SimpleNamespace(config=SimpleNamespace(representation_dim=2)),
                                  device=torch.device('cpu'))
            summary = {'latent_whitening': {'enabled': True, 'file': 'latent_whitening.pt'}}
            restore_whitening(run, summary, directory)
            x = torch.tensor([[4., 1.]])
            torch.testing.assert_close(run.latent_whitening(x), transform(x))
            with self.assertRaises(ValueError):
                BackgroundWhitening.load(directory / 'latent_whitening.pt', input_dim=3, device='cpu')
            (directory / 'latent_whitening.pt').unlink()
            with self.assertRaises(FileNotFoundError):
                restore_whitening(run, summary, directory)
            restore_whitening(SimpleNamespace(), {}, directory)  # Legacy, unwhitened run.

    def test_fit_uses_train_background_only_and_caps_events(self):
        loader = Mock(return_value=[raw_batch([1., 2., 3.]), raw_batch([4., 5., 6.])])
        run = SimpleNamespace(summary={}, loader=loader, backgrounds=['label_Wqq'],
                              signals=['label_Hbb'], num_workers=0, seed=42,
                              checkpoint=Path('backbone.pth'), device=torch.device('cpu'),
                              precision='fp32', encode=lambda b: b['x_particles'][:, 0])
        transform, metadata = fit_background_whitening(run, num_events=5, batch_size=3, eps=1e-5)
        self.assertEqual(loader.call_args.args, ('train', ['label_Wqq']))
        self.assertFalse(loader.call_args.kwargs['infinite'])
        self.assertEqual(metadata['num_events'], 5)
        torch.testing.assert_close(transform.mean, torch.tensor([3.]))
        torch.testing.assert_close(transform.matrix, torch.tensor([[1 / (2.5 ** .5)]]))

    def test_shared_transform_applies_to_signal_background_and_cache_once(self):
        class Head(torch.nn.Module):
            def forward_logits(self, z):
                return z[:, 0]
        run = SimpleNamespace(device=torch.device('cpu'), precision='fp32',
                              encode=Mock(side_effect=lambda b: b['x_particles'][:, 0]),
                              latent_whitening=BackgroundWhitening(torch.tensor([2.]), torch.tensor([[3.]])))
        bg, sg = raw_batch([1., 2., 3.]), raw_batch([4.])
        torch.testing.assert_close(encode_cwola(run, sg), torch.tensor([[6.]]))
        cache = []
        result = evaluate(Head(), run, ([bg], [sg]), 1, cache=cache)
        expected = torch.sigmoid(torch.tensor([-3., 0., 3., 6.])).numpy()
        np.testing.assert_allclose(result['scores'], expected)
        run.encode.side_effect = AssertionError('Cache must avoid backbone and repeated whitening')
        again = evaluate(Head(), run, (None, None), 1, cache=cache)
        np.testing.assert_array_equal(again['scores'], result['scores'])


if __name__ == '__main__':
    unittest.main()
