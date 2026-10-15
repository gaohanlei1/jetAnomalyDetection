"""Momentum-only JetCLR physics, objective and existing-loop contract tests."""
import math
import unittest
from unittest.mock import patch
import tempfile
from pathlib import Path
from dataclasses import replace

import torch
import torch.nn.functional as F

from models.part_jetclass import (
    ParticleTransformerConfig, MultiViewAugmentationConfig,
    CorruptedNegativeAugmentationConfig, JetCLRAugmentationConfig,
    JetCLRAugmentation, JetCLRContrastiveLoss,
    JetCLRParticleTransformerRepresentation,
)


def ddp_loss_worker(rank, store_path, result_dir):
    torch.set_num_threads(1)
    torch.distributed.init_process_group('gloo', init_method='file://' + store_path,
                                         rank=rank, world_size=2)
    try:
        torch.manual_seed(7)
        full = torch.randn(3, 4, 5)
        local = full[:, rank*2:(rank+1)*2].clone().requires_grad_()
        loss = JetCLRContrastiveLoss()(local)
        loss.backward()
        torch.save((loss.detach(), local.grad), Path(result_dir) / f'{rank}.pt')
    finally:
        torch.distributed.destroy_process_group()


def model_and_input(cms=False):
    config = ParticleTransformerConfig(embed_dim=8, num_heads=2, num_layers=1,
                                      num_class_layers=1, representation_dim=6,
                                      dropout=0, pairwise_hidden_dim=8)
    if cms:
        names = list(config.input_feature_names)
        names[8:10] = ['Cpfcan_dxysig', 'Cpfcan_dzsig']
        config = replace(config, input_feature_names=tuple(names),
                         standardized_feature_names=('log_pt_fraction', *names[8:10]))
    model = JetCLRParticleTransformerRepresentation(
        config, MultiViewAugmentationConfig(num_global_views=2, num_local_views=1),
        negative_augmentation_config=CorruptedNegativeAugmentationConfig(num_negative_views=9))
    eta = torch.tensor([[0.5, 0.8, 0.0], [-0.6, -0.2, -0.4]])
    phi = torch.tensor([[2.9, -3.0, 0.0], [-1.0, -1.2, -0.8]])
    pt = torch.tensor([[30., 20., 0.], [40., 10., 5.]])
    x = torch.randn(2, 3, 16)
    x[..., 0] = pt * phi.cos()
    x[..., 1] = pt * phi.sin()
    x[..., 2] = pt * eta.sinh()
    x[..., 3] = pt * eta.cosh()
    mask = pt == 0
    x[mask] = 0
    return model, x, mask, torch.stack((eta, phi, pt), -1)


class JetCLRTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        torch.set_num_threads(1)

    def test_absolute_coordinates_and_redundant_features_ignored(self):
        for cms in (False, True):
            model, x, mask, expected = model_and_input(cms)
            k, _ = model._extract_kinematics(x, mask)
            torch.testing.assert_close(k, expected)
            model.eval()
            reference = model.forward_representation(x, mask)
            altered = x.clone()
            altered[..., 3:] = float('nan')  # Includes energy and all extra features.
            torch.testing.assert_close(reference, model.forward_representation(altered, mask))
            torch.testing.assert_close(reference, model.representation_head(model(x, mask)))

    def test_splitting_conserves_four_momentum_without_truncation(self):
        _, _, mask, k = model_and_input()
        config = JetCLRAugmentationConfig(distortion_scale=0, split_probability=1,
                                         max_splits=3, eta_shift_max=0, phi_shift_max=0)
        aug, aug_mask = JetCLRAugmentation(config)(k, mask)
        self.assertEqual(aug.shape[1], 6)
        torch.testing.assert_close((~aug_mask).sum(1), 2 * (~mask).sum(1))

        def sum_p4(a):
            eta, phi, pt = a.unbind(-1)
            return torch.stack((pt*phi.cos(), pt*phi.sin(), pt*eta.sinh(), pt*eta.cosh()), -1).sum(1)

        torch.testing.assert_close(sum_p4(aug), sum_p4(k))
        self.assertTrue((aug[aug_mask] == 0).all())

    def test_common_shift_and_periodic_phi(self):
        _, _, mask, k = model_and_input()
        aug, out_mask = JetCLRAugmentation(JetCLRAugmentationConfig(
            distortion_scale=0, max_splits=0))(k, mask)
        torch.testing.assert_close(out_mask, mask)
        torch.testing.assert_close(aug[..., 2], k[..., 2])
        self.assertTrue(((aug[..., 1] >= -math.pi) & (aug[..., 1] < math.pi)).all())
        for i in range(2):
            valid = ~mask[i]
            delta = aug[i, valid, :2] - k[i, valid, :2]
            delta[:, 1] = torch.atan2(delta[:, 1].sin(), delta[:, 1].cos())
            torch.testing.assert_close(delta, delta[:1].expand_as(delta))

    def test_distortion_inverse_absolute_pt(self):
        k = torch.zeros(30000, 2, 3)
        k[..., 2] = torch.tensor([2., 20.])
        mask = torch.zeros(k.shape[:2], dtype=torch.bool)
        out, _ = JetCLRAugmentation(JetCLRAugmentationConfig(
            distortion_scale=0.1, max_splits=0, eta_shift_max=0, phi_shift_max=0))(k, mask)
        std = out[..., 0].std(0)
        torch.testing.assert_close(std, torch.tensor([0.05, 0.005]), rtol=0.03, atol=0)

    def test_loss_matches_explicit_equation_and_gradient(self):
        for views in (2, 3):
            z = torch.randn(views, 3, 5, requires_grad=True)
            actual = JetCLRContrastiveLoss(0.1)(z)
            unit = F.normalize(z, dim=-1)
            losses = []
            for v in range(views):
                for j in range(3):
                    negative = torch.stack([unit[v,j] @ unit[w,k] / .1
                                            for w in range(views) for k in range(3) if k != j])
                    for w in range(views):
                        if w != v:
                            losses.append(negative.logsumexp(0) - unit[v,j] @ unit[w,j] / .1)
            expected = torch.stack(losses).mean()
            torch.testing.assert_close(actual, expected)
            torch.testing.assert_close(torch.autograd.grad(actual, z, retain_graph=True)[0],
                                       torch.autograd.grad(expected, z)[0])

    def test_training_contract_views_backward_and_restore(self):
        model, x, mask, _ = model_and_input()
        out = model.forward_pretrain(x, torch.zeros(2, 10), mask, return_views=True)
        self.assertEqual(out['z_views'].shape, (3, 2, 6))
        self.assertEqual(out['z_negatives'].shape, (0, 2, 6))
        self.assertEqual(out['view_types'], ['global', 'global', 'local'])
        self.assertEqual(out['negative_views'], [])
        out['total_loss'].backward()
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        other, _, _, _ = model_and_input(cms=True)
        other.load_state_dict(model.state_dict(), strict=True)
        model.eval()
        other.eval()
        torch.testing.assert_close(model.forward_representation(x, mask),
                                   other.forward_representation(x, mask))

    def test_first_view_is_original_and_only_remaining_views_are_augmented(self):
        for global_views, local_views, max_splits in ((2, 0, 3), (1, 2, 3), (0, 2, 0)):
            model, x, mask, expected = model_and_input()
            model.num_global_views = global_views
            model.num_local_views = local_views
            model.augmentation.config.max_splits = max_splits
            model.augmentation.config.split_probability = 1
            original_input = x.clone()
            with patch.object(model.augmentation, 'forward',
                              wraps=model.augmentation.forward) as augment:
                out = model.forward_pretrain(x, padding_mask=mask, return_views=True)
            self.assertEqual(augment.call_count, model.num_views - 1)
            self.assertEqual(len(out['views']), model.num_views)
            torch.testing.assert_close(x, original_input)
            first, first_mask = out['views'][0], out['view_padding_masks'][0]
            torch.testing.assert_close(first[:, :x.shape[1]], expected)
            torch.testing.assert_close(first_mask[:, :x.shape[1]], mask)
            self.assertTrue((first[:, x.shape[1]:] == 0).all())
            self.assertTrue(first_mask[:, x.shape[1]:].all())
            self.assertTrue(all(v.shape == first.shape for v in out['views']))
            self.assertTrue(torch.isfinite(out['total_loss']))

    def test_no_negative_batch_and_invalid_input_fail_clearly(self):
        with self.assertRaisesRegex(ValueError, 'two samples'):
            JetCLRContrastiveLoss()(torch.randn(2, 1, 5))
        model, x, mask, _ = model_and_input()
        with self.assertRaisesRegex(ValueError, 'empty jets'):
            model(x, torch.ones_like(mask))
        with self.assertRaisesRegex(ValueError, 'positive pt'):
            x[0, 0, :3] = 0
            model(x, mask)

    def test_distributed_loss_and_gradient_match_global_batch(self):
        import torch.multiprocessing as mp
        with tempfile.TemporaryDirectory() as folder:
            mp.spawn(ddp_loss_worker, args=(folder + '/store', folder), nprocs=2, join=True)
            results = [torch.load(Path(folder) / f'{rank}.pt') for rank in range(2)]
        torch.manual_seed(7)
        full = torch.randn(3, 4, 5, requires_grad=True)
        expected = JetCLRContrastiveLoss()(full)
        expected.backward()
        torch.testing.assert_close(torch.stack([r[0] for r in results]).mean(), expected)
        # Parameter gradients are subsequently averaged by DDP.
        actual_grad = torch.cat([r[1] for r in results], dim=1) / 2
        torch.testing.assert_close(actual_grad, full.grad)


if __name__ == '__main__':
    unittest.main()
