def assert_within_se(estimate: float, reference: float, stderr: float, k: float = 4.0) -> None:
    """Fail unless ``|estimate - reference| < k * stderr`` (seeded MC checks)."""

    scale = max(stderr, 1e-15)
    assert abs(estimate - reference) < k * scale, (
        f"|{estimate} - {reference}| = {abs(estimate - reference)} "
        f"not < {k} * stderr ({k * scale})"
    )
