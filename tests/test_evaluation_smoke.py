import pytest


def test_frame_prf_and_summary_helpers():
    pytest.importorskip("mir_eval")
    pytest.importorskip("pretty_midi")
    import numpy as np
    import pandas as pd

    from piano_modeling.evaluation import frame_prf, summarize_evaluation_report

    metrics = frame_prf(np.array([1, 1, 0, 0]), np.array([1, 0, 1, 0]))
    assert 0.0 <= metrics["precision"] <= 1.0
    assert 0.0 <= metrics["recall"] <= 1.0
    assert 0.0 <= metrics["f1"] <= 1.0

    report = pd.DataFrame(
        [
            {
                "note_onset_f1": 0.25,
                "note_onset_offset_f1": 0.2,
                "velocity_f1": 0.1,
                "pedal_f1": 0.5,
                "num_pred_notes": 10,
                "num_ref_notes": 12,
            },
            {
                "note_onset_f1": 0.75,
                "note_onset_offset_f1": 0.4,
                "velocity_f1": 0.3,
                "pedal_f1": 0.7,
                "num_pred_notes": 14,
                "num_ref_notes": 12,
            },
        ]
    )
    summary = summarize_evaluation_report(report)
    assert summary["note_onset_f1"] == 0.5
    assert summary["num_ref_notes"] == 12.0
