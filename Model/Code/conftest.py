# conftest.py
import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--model",
        action="store",
        default="both",
        choices=["unimamba", "bimamba", "latentmoe", "both"],
        help="Which model to test: unimamba | bimamba | latentmoe | both (default: both)",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "unimamba: mark test as UniMamba-only")
    config.addinivalue_line("markers", "bimamba: mark test as BiMamba-only")
    config.addinivalue_line("markers", "latentmoe: mark test as LatentMoE-only")


def pytest_collection_modifyitems(config, items):
    choice = config.getoption("--model")
    skip_uni = pytest.mark.skip(reason="--model flag excluded unimamba")
    skip_bi  = pytest.mark.skip(reason="--model flag excluded bimamba")
    skip_moe = pytest.mark.skip(reason="--model flag excluded latentmoe")

    for item in items:
        if choice == "unimamba":
            if "bimamba"   in item.keywords: item.add_marker(skip_bi)
            if "latentmoe" in item.keywords: item.add_marker(skip_moe)
        elif choice == "bimamba":
            if "unimamba"  in item.keywords: item.add_marker(skip_uni)
            if "latentmoe" in item.keywords: item.add_marker(skip_moe)
        elif choice == "latentmoe":
            if "unimamba"  in item.keywords: item.add_marker(skip_uni)
            if "bimamba"   in item.keywords: item.add_marker(skip_bi)