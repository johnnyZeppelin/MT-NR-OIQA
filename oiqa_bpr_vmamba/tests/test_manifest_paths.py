from __future__ import annotations

from pathlib import Path

import pandas as pd

from oiqa_bpr_vmamba.cli.build_cviq_manifest import (
    _make_degraded_viewport,
    _make_restored_viewport,
    _normalize_path,
)


def test_path_prefix_resolution() -> None:
    prefix = Path('/tmp/prefix')
    rel = 'data/CVIQ/view_ports/326/326_fov15.png'
    assert _normalize_path(rel, prefix) == prefix / rel


def test_viewport_path_mirroring() -> None:
    src = Path('/tmp/prefix/data/CVIQ/view_ports/326/326_fov15.png')
    restored_root = Path('/tmp/prefix/data/CVIQ/restored/view_ports_restored')
    degraded_root = Path('/tmp/prefix/data/CVIQ/degraded/view_ports_degraded')
    assert _make_restored_viewport(src, restored_root) == restored_root / '326' / '326_fov15_r.png'
    assert _make_degraded_viewport(src, degraded_root) == degraded_root / '326' / '326_fov15_d.png'


def test_viewport_path_mirroring_with_compression_folder() -> None:
    src = Path('/tmp/prefix/data/CVIQ/view_ports/JPEG/326_fov15.png')
    restored_root = Path('/tmp/prefix/data/CVIQ/restored/view_ports_restored')
    assert _make_restored_viewport(src, restored_root) == restored_root / 'JPEG' / '326_fov15_r.png'


from oiqa_bpr_vmamba.utils.splits import create_or_load_splits
from oiqa_bpr_vmamba.utils.io import save_json


def test_create_or_load_splits_changes_with_ratios(tmp_path: Path) -> None:
    manifest = tmp_path / 'manifest.csv'
    pd.DataFrame({
        'image_id': [str(i) for i in range(12)],
        'compression_type': ['JPEG'] * 12,
    }).to_csv(manifest, index=False)
    base = {
        'paths': {'manifest_csv': str(manifest), 'split_dir': str(tmp_path / 'splits')},
        'split': {'train_ratio': 0.8, 'val_ratio': 0.1, 'test_ratio': 0.1, 'stratify_by': None, 'split_seed': 1},
    }
    train_a, val_a, test_a = create_or_load_splits(base)
    assert train_a.name != ''
    other = {
        'paths': base['paths'],
        'split': {'train_ratio': 0.5, 'val_ratio': 0.25, 'test_ratio': 0.25, 'stratify_by': None, 'split_seed': 1},
    }
    train_b, val_b, test_b = create_or_load_splits(other)
    assert train_a != train_b
    assert val_a != val_b
    assert test_a != test_b


def test_create_or_load_splits_supports_zero_val_or_test(tmp_path: Path) -> None:
    manifest = tmp_path / 'manifest.csv'
    pd.DataFrame({
        'image_id': [str(i) for i in range(10)],
        'compression_type': ['JPEG'] * 10,
    }).to_csv(manifest, index=False)
    cfg = {
        'paths': {'manifest_csv': str(manifest), 'split_dir': str(tmp_path / 'splits_zero')},
        'split': {'train_ratio': 0.8, 'val_ratio': 0.0, 'test_ratio': 0.2, 'stratify_by': None, 'split_seed': 7},
    }
    train_csv, val_csv, test_csv = create_or_load_splits(cfg)
    assert len(pd.read_csv(train_csv)) == 8
    assert len(pd.read_csv(val_csv)) == 0
    assert len(pd.read_csv(test_csv)) == 2
