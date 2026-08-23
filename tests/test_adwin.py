from app.drift.adwin import ADWINConceptDriftDetector


def test_adwin_eventually_detects_error_shift():
    detector = ADWINConceptDriftDetector(delta=0.01, min_samples=30)
    events = []
    for idx in range(240):
        prediction = 0
        truth = 0 if idx < 120 else 1
        event = detector.update(prediction, truth, idx)
        if event is not None:
            events.append(event)

    assert events
    assert events[0].drift_detected


def test_adwin_skips_unlabeled_records():
    detector = ADWINConceptDriftDetector(min_samples=5)
    assert detector.update(1, None, 0) is None
    assert detector.labeled_samples == 0
