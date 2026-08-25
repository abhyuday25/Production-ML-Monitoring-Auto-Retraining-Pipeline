from pathlib import Path

from scripts import ensure_model as ensure_model_module


def test_ensure_model_retrains_when_champion_alias_is_not_loadable(monkeypatch, tmp_path: Path):
    calls = {"prepare": 0, "train": 0, "load_checks": 0}

    def fake_champion_is_loadable():
        calls["load_checks"] += 1
        return calls["load_checks"] > 1

    def fake_prepare(data_dir, rows, random_seed):
        calls["prepare"] += 1

    def fake_train(data_dir):
        calls["train"] += 1
        return {"model_version": "7"}

    monkeypatch.setattr(ensure_model_module, "champion_is_loadable", fake_champion_is_loadable)
    monkeypatch.setattr(ensure_model_module, "prepare_datasets", fake_prepare)
    monkeypatch.setattr(ensure_model_module, "train_model", fake_train)

    result = ensure_model_module.ensure_model(tmp_path / "data", rows=100, random_seed=42, force=False)

    assert result == {"status": "trained", "model_version": "7"}
    assert calls == {"prepare": 1, "train": 1, "load_checks": 2}
