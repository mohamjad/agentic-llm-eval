"""Tests for configuration persistence and merge behavior."""

import json

from src.utils.config import Config, get_default_config, load_config, save_config


def test_config_save_and_load_json_and_yaml(tmp_path):
    config = Config()
    config.set("evaluator.success_threshold", 0.82)
    config.set("logging.level", "DEBUG")

    json_path = tmp_path / "config.json"
    yaml_path = tmp_path / "config.yaml"
    save_config(config, str(json_path))
    config.save_to_file(yaml_path, format="yaml")

    loaded_json = Config().load_from_file(json_path)
    loaded_yaml = Config().load_from_file(yaml_path)

    assert loaded_json.get("evaluator.success_threshold") == 0.82
    assert loaded_yaml.get("logging.level") == "DEBUG"


def test_config_load_from_file_replace_and_required_get(tmp_path):
    payload = {"evaluator": {"success_threshold": 0.55}}
    file_path = tmp_path / "replace.json"
    file_path.write_text(json.dumps(payload), encoding="utf-8")

    config = Config().load_from_file(file_path, merge=False)

    assert config.get("evaluator.success_threshold", required=True) == 0.55
    assert config.get("metrics.efficiency.max_reasonable_steps") is None


def test_get_default_config_and_load_config_use_default_locations(tmp_path, monkeypatch):
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    (config_dir / "config.json").write_text(
        json.dumps({"evaluator": {"success_threshold": 0.73}}),
        encoding="utf-8",
    )

    monkeypatch.chdir(tmp_path)
    from src.utils import config as config_module

    config_module._default_config = None
    default_config = get_default_config()
    loaded = load_config()

    assert default_config.get("evaluator.success_threshold") == 0.73
    assert loaded.get("evaluator.success_threshold") == 0.73
