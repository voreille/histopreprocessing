import pytest
from unittest.mock import patch, MagicMock

from histopreprocessing.histoqc import run_histoqc_task, parse_histoqc_config_mapping


@pytest.fixture
def fake_wsi_dir(tmp_path):
    """Create a fake input WSI directory with test files."""
    wsi_dir = tmp_path / "wsi_data"
    wsi_dir.mkdir()
    (wsi_dir / "TCGA-AB-1234.svs").touch()
    (wsi_dir / "TCGA-XY-5678.svs").touch()
    return wsi_dir


@pytest.fixture
def fake_output_dir(tmp_path):
    """Create a fake output directory."""
    output_dir = tmp_path / "histoqc_output"
    output_dir.mkdir()
    return output_dir


@pytest.fixture
def fake_config(tmp_path):
    """Create a fake HistoQC YAML config."""
    config_path = tmp_path / "histoqc_config.yaml"
    config_path.write_text("""
    mappings:
      - config: /configs/default.yaml
        wsi_ids:
          - TCGA-AB-1234
          - TCGA-XY-5678
    """)
    return config_path


###  Test: Parsing HistoQC Config
def test_parse_histoqc_config_mapping(fake_wsi_dir, fake_config):
    """Test that WSI files are correctly matched using a HistoQC mapping file."""

    def mock_mapping_func(filename):
        return filename.stem  # Extract WSI ID from filename

    file_list, config_list = parse_histoqc_config_mapping(
        fake_config, fake_wsi_dir, "*.svs", mock_mapping_func)

    assert len(file_list) == 2
    assert len(config_list) == 2
    assert "TCGA-AB-1234.svs" in str(file_list[0])
    assert "TCGA-XY-5678.svs" in str(file_list[1])


###  Test: Running HistoQC with Mocks
@patch("subprocess.run")
def test_run_histoqc_task(mock_subprocess, fake_wsi_dir, fake_output_dir,
                          fake_config):
    """Test that run_histoqc_task calls Docker with correct arguments."""

    mock_subprocess.return_value = MagicMock(returncode=0)

    run_histoqc_task(input_dir=fake_wsi_dir,
                     output_dir=fake_output_dir,
                     config=fake_config,
                     wsi_id_mapping_style="TCGA")

    # Assert Docker was called
    mock_subprocess.assert_called_once()
    docker_cmd = mock_subprocess.call_args[0][
        0]  # Get the first call arguments

    assert "docker" in docker_cmd
    assert "run" in docker_cmd
    assert "histotools/histoqc:master" in docker_cmd  # Expected Docker image
    assert "/data/output" in docker_cmd  # Expected output dir inside container


### 🔹 Test: Handling Missing Files
@patch("subprocess.run")
def test_run_histoqc_task_no_files(mock_subprocess, fake_output_dir,
                                   fake_config):
    """Test behavior when no WSI files are found."""
    empty_dir = fake_output_dir / "empty_input"
    empty_dir.mkdir()

    with pytest.raises(ValueError, match="No files found"):
        run_histoqc_task(input_dir=empty_dir,
                         output_dir=fake_output_dir,
                         config=fake_config,
                         wsi_id_mapping_style="TCGA")

    # Docker should NOT be called if no files exist
    mock_subprocess.assert_not_called()
