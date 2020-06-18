from math import isclose

import pytest

from lhotse.cut import Cut
from lhotse.features import Features
from lhotse.supervision import SupervisionSegment


@pytest.fixture
def overlapping_supervisions_cut():
    return Cut(
        id='cut-1',
        start=0.0,
        duration=0.5,
        features=Features(
            recording_id='recording-1',
            channel_id=0,
            start=0,
            duration=0.5,
            frame_length=25.0,
            frame_shift=10.0,
            storage_type='lilcom',
            storage_path='test/fixtures/dummy_feats/storage/e66b6386-aee5-4a5a-8369-fdde1d2b97c7.llc'
        ),
        supervisions=[
            SupervisionSegment(id='s1', recording_id='recording-1', start=0.0, duration=0.2),
            SupervisionSegment(id='s2', recording_id='recording-1', start=0.1, duration=0.2),
            SupervisionSegment(id='s3', recording_id='recording-1', start=0.2, duration=0.2),
            SupervisionSegment(id='s4', recording_id='recording-1', start=0.3, duration=0.2)
        ]
    )


@pytest.mark.parametrize(
    ['offset', 'until', 'keep_excessive_supervisions', 'expected_supervision_ids'],
    [
        (0.0, None, True, ['s1', 's2', 's3', 's4']),
        (0.0, None, False, ['s1', 's2', 's3', 's4']),
        (0.0, 0.5, True, ['s1', 's2', 's3', 's4']),
        (0.0, 0.5, False, ['s1', 's2', 's3', 's4']),
        (0.1, None, True, ['s1', 's2', 's3', 's4']),
        (0.1, None, False, ['s2', 's3', 's4']),
        (0.0, 0.4, True, ['s1', 's2', 's3', 's4']),
        (0.0, 0.4, False, ['s1', 's2', 's3']),
        (0.1, 0.4, True, ['s1', 's2', 's3', 's4']),
        (0.1, 0.4, False, ['s2', 's3']),
        (0.1, 0.2, True, ['s1', 's2']),
        (0.1, 0.2, False, []),
        (0.2, None, True, ['s2', 's3', 's4']),
        (0.2, None, False, ['s3', 's4']),
        (0.2, 0.4, True, ['s2', 's3', 's4']),
        (0.2, 0.4, False, ['s3']),
        (0.0, 0.1, True, ['s1']),
        (0.0, 0.1, False, []),
        (0.1, 0.2, False, []),
        (0.2, 0.3, False, []),
        (0.3, 0.4, False, []),
        (0.4, 0.5, False, []),
        (0.27, 0.31, False, []),
    ]
)
def test_truncate_cut(
        offset,
        until,
        keep_excessive_supervisions,
        expected_supervision_ids,
        overlapping_supervisions_cut
):
    truncated_cut = overlapping_supervisions_cut.truncate(
        offset=offset,
        until=until,
        keep_excessive_supervisions=keep_excessive_supervisions
    )
    remaining_supervision_ids = [s.id for s in truncated_cut.supervisions]
    assert remaining_supervision_ids == expected_supervision_ids


@pytest.mark.parametrize(
    ['offset', 'until', 'keep_excessive_supervisions', 'expected_supervision_ids'],
    [
        (0.0, None, True, ['s1', 's2', 's3', 's4']),
    ]
)
def test_truncate_mixed_cut(
        offset,
        until,
        keep_excessive_supervisions,
        expected_supervision_ids,
        cut1,
        cut2,
        mixed_cut,
        cut_set
):
    truncated_cut = mixed_cut.truncate(
        offset=offset,
        until=until,
        keep_excessive_supervisions=keep_excessive_supervisions
    )
    truncated_cut.with_cut_set(cut_set)
    remaining_supervision_ids = [s.id for s in truncated_cut.supervisions]
    assert remaining_supervision_ids == [s.id for s in cut1.supervisions + cut2.supervisions]


def test_truncate_cut_set_offset_start(cut_set):
    truncated_cut_set = cut_set.truncate(max_duration=5, offset_type='start')
    for cut in truncated_cut_set:
        assert isclose(cut.duration, 5.0)
        assert isclose(cut.start, 0.0)


def test_truncate_cut_set_offset_end(cut_set):
    truncated_cut_set = cut_set.truncate(max_duration=5, offset_type='end')
    for cut in truncated_cut_set:
        assert isclose(cut.duration, 5.0)
        assert isclose(cut.start, 5.0)


def test_truncate_cut_set_offset_random(cut_set):
    truncated_cut_set = cut_set.truncate(max_duration=5, offset_type='random')
    for cut in truncated_cut_set:
        assert isclose(cut.duration, 5.0)
        assert 0.0 <= cut.start <= 5.0
    # Check that "cut.start" is not the same in every cut
    assert len(set(cut.start for cut in truncated_cut_set)) > 1


def test_truncate_cut_set_offset_start(cut_set):
    truncated_cut_set = cut_set.truncate(max_duration=5, offset_type='start')
    for cut in truncated_cut_set:
        assert isclose(cut.duration, 5.0)
        assert isclose(cut.start, 0.0)
