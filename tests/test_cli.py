import re

from monte_carlo_option_engine.cli import main

_SYNTH = [
    "--S",
    "100",
    "--K",
    "105",
    "--T",
    "0.5",
    "--r",
    "0.04",
    "--q",
    "0.01",
    "--sigma",
    "0.25",
    "--n",
    "2000",
    "--steps",
    "20",
    "--seed",
    "0",
]


def _price(text: str) -> float:
    match = re.search(r"price=([0-9,]+\.[0-9]+)", text)
    assert match is not None, text
    return float(match.group(1).replace(",", ""))


def test_cli_euro_call_prints_price_and_bs(capsys) -> None:
    code = main(["price", "--kind", "euro_call", *_SYNTH])
    captured = capsys.readouterr()
    assert code == 0
    assert "BS euro_call:" in captured.out
    price = _price(captured.out)
    assert price > 0.0
    bs_match = re.search(r"BS euro_call: ([0-9,]+\.[0-9]+)", captured.out)
    assert bs_match is not None
    bs = float(bs_match.group(1).replace(",", ""))
    assert abs(price - bs) < 1e-8


def test_cli_moneyness_matches_strike(capsys) -> None:
    args = [
        "price",
        "--kind",
        "euro_call",
        "--S",
        "100",
        "--moneyness",
        "1.05",
        "--T",
        "0.5",
        "--r",
        "0.04",
        "--q",
        "0.01",
        "--sigma",
        "0.25",
        "--n",
        "2000",
        "--seed",
        "0",
    ]
    assert main(args) == 0
    price = _price(capsys.readouterr().out)
    assert price > 0.0


def test_cli_asian_omits_bs_line(capsys) -> None:
    code = main(["price", "--kind", "asian_call", *_SYNTH])
    captured = capsys.readouterr()
    assert code == 0
    assert "BS " not in captured.out
    assert _price(captured.out) >= 0.0


def test_cli_barrier_requires_B(capsys) -> None:
    code = main(["price", "--kind", "up_and_out_call", *_SYNTH])
    captured = capsys.readouterr()
    assert code == 2
    assert "--B is required" in captured.err


def test_cli_barrier_prices(capsys) -> None:
    code = main(["price", "--kind", "up_and_out_call", "--B", "130", *_SYNTH])
    captured = capsys.readouterr()
    assert code == 0
    assert _price(captured.out) >= 0.0
    assert "BS " not in captured.out


def test_cli_missing_market_returns_error(capsys) -> None:
    code = main(
        [
            "price",
            "--kind",
            "euro_call",
            "--K",
            "105",
            "--T",
            "0.5",
            "--r",
            "0.04",
        ]
    )
    captured = capsys.readouterr()
    assert code == 2
    assert "pass --S and --sigma" in captured.err
