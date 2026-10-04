"""Các kiểm tra nhỏ cho những phần dễ sai; chạy bằng python -m unittest test_code -v."""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from torch import nn

import inference
import dataset
import losses
import model


class TestLosses(unittest.TestCase):
    def test_focal_gamma_zero_equals_ce(self):
        torch.manual_seed(1)
        logits = torch.randn(8, 9)
        labels = torch.randint(0, 9, (8,))
        got = losses.FocalLoss(gamma=0)(logits, labels)
        expected = nn.CrossEntropyLoss()(logits, labels)
        torch.testing.assert_close(got, expected, atol=1e-6, rtol=0)

    def test_cutmix_lambda_matches_actual_area(self):
        x = torch.stack([torch.zeros(3, 8, 8), torch.ones(3, 8, 8)])
        y = torch.tensor([0, 1])
        with patch("numpy.random.beta", return_value=0.5), \
             patch("torch.randperm", return_value=torch.tensor([1, 0])), \
             patch("torch.randint", return_value=torch.tensor([4])):
            mixed, (ya, yb, lam) = losses.mix_batch(x, y, mode="cutmix")
        self.assertEqual(ya.tolist(), [0, 1])
        self.assertEqual(yb.tolist(), [1, 0])
        changed_fraction = float((mixed[0, 0] != x[0, 0]).float().mean())
        self.assertAlmostEqual(lam, 1 - changed_fraction)


class TestDataset(unittest.TestCase):
    def test_load_original_two_column_subsets_and_validate_labels(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            pd.DataFrame({'Filename': ['a.jpg', 'b.jpg', 'c.jpg'],
                          'Label': [0, 1, 2], 'Species': ['A', 'B', 'C']}).to_csv(root / 'labels.csv', index=False)
            for part, name, label in [('train', 'a.jpg', 0), ('val', 'b.jpg', 1), ('test', 'c.jpg', 2)]:
                pd.DataFrame({'Filename': [name], 'Label': [label]}).to_csv(root / f'{part}_subset0.csv', index=False)
            train, val, test = dataset.load_split(root)
            self.assertEqual([frame.Species.iloc[0] for frame in (train, val, test)], ['A', 'B', 'C'])
            pd.DataFrame({'Filename': ['c.jpg'], 'Label': [1]}).to_csv(root / 'test_subset0.csv', index=False)
            _, _, test = dataset.load_split(root)
            self.assertEqual((int(test.Label.iloc[0]), test.Species.iloc[0]), (1, 'B'))
            pd.DataFrame({'Filename': ['missing.jpg'], 'Label': [1]}).to_csv(root / 'test_subset0.csv', index=False)
            with self.assertRaisesRegex(ValueError, 'thiếu trong labels.csv'):
                dataset.load_split(root)


class TestInference(unittest.TestCase):
    def test_fuse_bn_preserves_eval_outputs(self):
        torch.manual_seed(2)
        net = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.BatchNorm2d(4)).eval()
        x = torch.randn(2, 3, 12, 12)
        fused = inference.fuse_conv_bn(net)
        self.assertIsInstance(fused[1], nn.Identity)
        with torch.inference_mode():
            torch.testing.assert_close(fused(x), net(x), atol=1e-5, rtol=1e-5)

    def test_temperature_does_not_change_argmax(self):
        logits = np.array([[2., 1., -1.], [-2., 1., 0.]])
        for temp in (0.1, 1.0, 10.0):
            probs = inference.apply_temperature(logits, temp)
            np.testing.assert_array_equal(probs.argmax(1), logits.argmax(1))
            np.testing.assert_allclose(probs.sum(1), 1)


if __name__ == "__main__":
    unittest.main()
