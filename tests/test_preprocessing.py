import pytest
import torch
import json
from rlalloc.utils.preprocessing import TraceProcessor

@pytest.fixture
def sample_trace():
    return {
        "deviceProperties": [{
            "totalGlobalMem": 42297786368,
            "maxThreadsPerBlock": 1024
        }],
        "traceEvents": [
            {
                "ph": "X", "cat": "kernel",
                "ts": 5215175130088.848,
                "dur": 27863.606,
                "args": {"gpu_usage": 0.8}
            },
            {
                "ph": "X", "cat": "cpu_op",
                "ts": 5215175130088.848,
                "dur": 27863.606,
                "args": {"cpu_usage": 0.6}
            }
        ]
    }

@pytest.fixture
def processor(sample_trace, tmp_path):
    trace_file = tmp_path / "test_trace.json"
    with open(trace_file, "w") as f:
        json.dump(sample_trace, f)
    
    processor = TraceProcessor()
    processor.load_trace(str(trace_file))
    return processor

def test_trace_loading(processor):
    assert len(processor.kernel_events) > 0
    assert len(processor.cpu_events) > 0

def test_resource_usage(processor):
    usage = processor.get_resource_usage(5215175130088.848)
    assert 0 <= usage['gpu_util'] <= 1
    assert 0 <= usage['cpu_util'] <= 1
    assert 0 <= usage['mem_util'] <= 1

def test_time_window(processor):
    window = processor.get_time_window(5215175130088.848)
    assert isinstance(window, torch.Tensor)
    assert window.shape == (5, 20, 1)