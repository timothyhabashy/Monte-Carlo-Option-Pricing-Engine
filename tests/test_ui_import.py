import pytest

from monte_carlo_option_engine.ui import main


def test_ui_main_is_callable() -> None:
    assert callable(main)


def test_streamlit_extra_optional() -> None:
    pytest.importorskip("streamlit")
