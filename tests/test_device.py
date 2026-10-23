import pytest
import torch

from training.train import resolve_device, train


def test_cpu_device_is_explicitly_supported():
    assert resolve_device("cpu").type == "cpu"


def test_auto_resolves_to_available_execution_device():
    expected = "cuda" if torch.cuda.is_available() else "cpu"
    assert resolve_device("auto").type == expected


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_device_is_explicitly_supported():
    assert resolve_device("cuda").type == "cuda"


def test_unavailable_cuda_request_has_install_guidance():
    if torch.cuda.is_available():
        pytest.skip("CUDA is available")
    with pytest.raises(RuntimeError, match="CUDA-enabled PyTorch"):
        resolve_device("cuda")


def test_missing_dataset_explains_generation_command(tmp_path):
    with pytest.raises(FileNotFoundError, match="training.generate_data"):
        train(str(tmp_path / "absent.npz"), str(tmp_path / "models"), device="cpu")
