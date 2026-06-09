from omg.benchmarks.report import generation_benchmark_markdown


def test_generation_benchmark_markdown_bolds_best_by_direction():
    rows = [
        {"ckpt": "a", "FID": 1.0, "R@1": 0.2, "physical": {"mean": 2.0, "std": 0.1}},
        {"ckpt": "b", "FID": 0.5, "R@1": 0.3, "physical": {"mean": 3.0, "std": 0.2}},
    ]
    text = generation_benchmark_markdown(
        rows=rows,
        metric_directions={"FID": "min", "R@1": "max", "physical": "min"},
        metadata={"num_texts": 2},
    )

    assert "**0.5000**" in text
    assert "**0.3000**" in text
    assert "**2.0000 +/- 0.1000**" in text
