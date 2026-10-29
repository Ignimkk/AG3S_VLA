"""Meaningful fail-closed guards for historical source relinking."""
import importlib.util
import os
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    'retirement', Path(__file__).parents[1] / 'scripts/setup/plan_legacy_retirement.py')
retirement = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(retirement)


@pytest.fixture
def fixture(tmp_path):
    legacy, archive = tmp_path / 'work', tmp_path / 'archive'
    source, frozen = legacy / 'source', archive / 'source'
    source.mkdir(parents=True)
    frozen.mkdir(parents=True)
    for root in (source, frozen):
        (root / 'ignored-launcher.py').write_text('print(42)\n')
        (root / '.gitignore').write_text('ignored-launcher.py\n')
    incoming = tmp_path / 'historical' / 'source'
    incoming.parent.mkdir()
    incoming.symlink_to(source, target_is_directory=True)
    data = {'incoming_legacy_links': [{'path': str(incoming), 'target': str(source)}]}
    return data, legacy, archive, source, frozen, incoming


def build(fixture):
    data, legacy, archive, *_ = fixture
    return retirement.build_plan(data, archive, legacy=legacy, allowed_inputs=())


def test_equal_ignored_source_is_preserved_and_plan_is_read_only(fixture):
    plan = build(fixture)
    assert plan['ready_count'] == 1
    assert plan['blocked_count'] == 0
    assert 'ignored-launcher.py' in plan['subtrees']['source']['source_inventory']['entries']
    retirement.validate_plan_current(plan)
    assert os.readlink(fixture[-1]) == str(fixture[3])


def test_missing_ignored_source_blocks_migration(fixture):
    (fixture[4] / 'ignored-launcher.py').unlink()
    plan = build(fixture)
    assert plan['blocked_count'] == 1
    assert plan['subtrees']['source']['differences']['missing_from_archive'] == ['ignored-launcher.py']
    with pytest.raises(ValueError, match='blocked'):
        retirement.validate_plan_current(plan)


def test_changed_historical_content_rejected(fixture):
    (fixture[4] / 'ignored-launcher.py').write_text('print(43)\n')
    plan = build(fixture)
    assert plan['subtrees']['source']['differences']['different'] == ['ignored-launcher.py']


def test_runtime_exclusion_does_not_crawl_venv_or_git(fixture):
    for name in ('.venv', '.venv-old', '.cache', '__pycache__', '.git'):
        directory = fixture[3] / name
        directory.mkdir()
        (directory / 'ignored-launcher.py').write_text('not source')
    plan = build(fixture)
    assert plan['ready_count'] == 1
    assert len(plan['subtrees']['source']['source_inventory']['excluded_paths']) == 5


def test_symlink_escape_rejected_without_following(fixture, tmp_path):
    active = tmp_path / 'active-checkout'
    active.mkdir()
    (active / 'module.py').write_text('outside code')
    for root in (fixture[3], fixture[4]):
        (root / 'escape').symlink_to(active, target_is_directory=True)
    plan = build(fixture)
    assert plan['blocked_count'] == 1
    for side in ('source_inventory', 'archive_inventory'):
        inventory = plan['subtrees']['source'][side]
        assert inventory['entries']['escape']['kind'] == 'symlink'
        assert 'escape/module.py' not in inventory['entries']
        assert any('escapes' in issue['reason'] for issue in inventory['issues'])


def test_absolute_archive_link_back_to_legacy_is_rejected(fixture):
    (fixture[3] / 'internal').symlink_to(fixture[3] / 'ignored-launcher.py')
    (fixture[4] / 'internal').symlink_to(fixture[3] / 'ignored-launcher.py')
    plan = build(fixture)
    assert plan['blocked_count'] == 1
    assert any('still depends' in x['reason']
               for x in plan['subtrees']['source']['differences']['archive_issues'])


def test_shared_input_link_supported_only_with_explicit_allowlist(fixture, tmp_path):
    shared = tmp_path / 'immutable-input'
    shared.mkdir()
    (shared / 'record.npy').write_bytes(b'input')
    for root in (fixture[3], fixture[4]):
        (root / 'input').symlink_to(shared / 'record.npy')
    assert build(fixture)['blocked_count'] == 1
    data, legacy, archive, *_ = fixture
    plan = retirement.build_plan(data, archive, legacy=legacy, allowed_inputs=(shared,))
    assert plan['ready_count'] == 1


def test_stale_plan_detects_content_and_incoming_target_changes(fixture):
    plan = build(fixture)
    for root in (fixture[3], fixture[4]):
        (root / 'ignored-launcher.py').write_text('print(44)\n')
    with pytest.raises(ValueError, match='changed after planning'):
        retirement.validate_plan_current(plan)
    fresh = build(fixture)
    fixture[-1].unlink()
    fixture[-1].symlink_to(fixture[4])
    with pytest.raises(ValueError, match='no longer safe'):
        retirement.validate_plan_current(fresh)


def test_duplicate_and_nonlegacy_targets_rejected(fixture):
    data, legacy, archive, *_ = fixture
    with pytest.raises(ValueError, match='duplicate'):
        retirement.build_plan({'incoming_legacy_links': data['incoming_legacy_links'] * 2},
                              archive, legacy=legacy)
    wrong = {'incoming_legacy_links': [{'path': str(fixture[-1]), 'target': str(archive)}]}
    with pytest.raises(ValueError, match='not a legacy subtree'):
        retirement.build_plan(wrong, archive, legacy=legacy)


def test_cross_subtree_source_link_needs_separate_identity_validation(fixture):
    legacy, archive = fixture[1:3]
    for root in (legacy, archive):
        adjacent = root / 'unverified-other-source'
        adjacent.mkdir()
        (adjacent / 'module.py').write_text('different identity')
        (root / 'source' / 'cross-source').symlink_to(adjacent)
    plan = build(fixture)
    assert plan['blocked_count'] == 1
    assert any('escapes' in issue['reason']
               for issue in plan['subtrees']['source']['differences']['source_issues'])


@pytest.fixture
def backlink_fixture(tmp_path):
    legacy = tmp_path / 'work'
    pi05 = legacy / 'pi05_TO_hybrid'
    pi05.mkdir(parents=True)
    storage = tmp_path / 'storage' / 'results' / 'rby1'
    storage.mkdir(parents=True)
    old = pi05 / 'outputs'
    old.symlink_to(storage, target_is_directory=True)
    incoming = tmp_path / 'sibling' / 'outputs'
    incoming.parent.mkdir()
    incoming.symlink_to(old, target_is_directory=True)
    rows = [{'path': str(incoming), 'target': str(old), 'resolved': str(storage)}]
    return rows, legacy, storage, old, incoming


def build_backlinks(fixture):
    rows, legacy, storage, *_ = fixture
    return retirement.build_backlink_plan(rows, legacy=legacy,
                                         allowed_targets={'outputs': storage})


def test_backlink_plan_preserves_physical_identity_without_hashing(backlink_fixture, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('shared storage must not be traversed or hashed')
    monkeypatch.setattr(retirement, 'sha256', forbidden)
    monkeypatch.setattr(os, 'scandir', forbidden)
    plan = build_backlinks(backlink_fixture)
    assert plan['ready_count'] == 1
    assert plan['replacements'][0]['new_target'] == str(backlink_fixture[2])
    retirement.validate_backlink_plan_current(plan)
    assert os.readlink(backlink_fixture[-1]) == str(backlink_fixture[-2])


def test_backlink_namespace_requires_exact_opt_in(backlink_fixture):
    rows, legacy, storage, *_ = backlink_fixture
    with pytest.raises(ValueError, match='opted-in'):
        retirement.build_backlink_plan(rows, legacy=legacy, allowed_targets={})
    with pytest.raises(ValueError, match='unsupported'):
        retirement.build_backlink_plan(rows, legacy=legacy, allowed_targets={'arbitrary': storage})


def test_backlink_chained_link_and_wrong_target_blocked(backlink_fixture, tmp_path):
    _, _, storage, old, _ = backlink_fixture
    intermediate = tmp_path / 'intermediate'
    intermediate.symlink_to(storage)
    old.unlink()
    old.symlink_to(intermediate)
    plan = build_backlinks(backlink_fixture)
    assert plan['blocked_count'] == 1
    assert any('directly' in issue for issue in plan['replacements'][0]['errors'])


def test_backlink_old_target_requires_symlink_and_existing_target(backlink_fixture):
    _, _, storage, old, _ = backlink_fixture
    old.unlink()
    old.mkdir()
    assert build_backlinks(backlink_fixture)['blocked_count'] == 1
    storage.rmdir()
    with pytest.raises(ValueError, match='existing physical'):
        build_backlinks(backlink_fixture)


def test_backlink_stale_legacy_inode_rejected_even_with_same_raw_target(backlink_fixture):
    plan = build_backlinks(backlink_fixture)
    _, _, storage, old, _ = backlink_fixture
    saved = old.with_name('saved-outputs')
    old.rename(saved)  # preserve inode, forcing replacement to have another inode
    old.symlink_to(storage)
    with pytest.raises(ValueError, match='changed after planning'):
        retirement.validate_backlink_plan_current(plan)


def test_backlink_stale_shared_target_inode_rejected(backlink_fixture):
    plan = build_backlinks(backlink_fixture)
    storage = backlink_fixture[2]
    storage.rename(storage.with_name('saved-rby1'))
    storage.mkdir()
    with pytest.raises(ValueError, match='changed after planning'):
        retirement.validate_backlink_plan_current(plan)


def test_backlink_stale_incoming_raw_target_rejected(backlink_fixture):
    plan = build_backlinks(backlink_fixture)
    incoming = backlink_fixture[-1]
    incoming.unlink()
    incoming.symlink_to(backlink_fixture[2])
    with pytest.raises(ValueError, match='no longer safe'):
        retirement.validate_backlink_plan_current(plan)
