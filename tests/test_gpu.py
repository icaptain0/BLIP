from BLIP_Video import get_gpu_usage

def test_get_gpu_usage_returns_str():
    result = get_gpu_usage()
    assert isinstance(result, str)
