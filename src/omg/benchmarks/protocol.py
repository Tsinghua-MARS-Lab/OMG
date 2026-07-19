from __future__ import annotations


BENCHMARK_COHORT_FAMILIES = {
    "text": {
        name: (name,)
        for name in (
            "100style",
            "amass",
            "bones_seed",
            "fitness",
            "humanml",
            "idea400",
            "lafan1",
            "motiongv",
            "motionllama",
            "omomo",
            "permo",
            "snapmogen",
        )
    },
    "audio": {
        name: (name,)
        for name in (
            "aioz_gdance",
            "aistpp",
            "compas3d",
            "finedance",
            "opendance",
        )
    },
    "humanref": {
        **{
            name: (name,)
            for name in (
                "aistpp",
                "amass",
                "finedance",
                "fitness",
                "humanml",
                "idea400",
                "motiongv",
                "motionllama",
                "permo",
                "snapmogen",
            )
        },
        "beat2": (
            "beat2_chinese",
            "beat2_english",
            "beat2_japanese",
            "beat2_spanish",
        ),
    },
}


def benchmark_condition_cohorts(condition: str, split: str) -> dict[str, tuple[str, ...]]:
    if condition not in BENCHMARK_COHORT_FAMILIES:
        raise ValueError(f"Unsupported benchmark condition {condition!r}")
    return {
        cohort: tuple(f"{family}_{split}" for family in families)
        for cohort, families in BENCHMARK_COHORT_FAMILIES[condition].items()
    }


def benchmark_source_datasets(condition: str, split: str) -> list[str]:
    return [
        source_dataset
        for source_datasets in benchmark_condition_cohorts(condition, split).values()
        for source_dataset in source_datasets
    ]
