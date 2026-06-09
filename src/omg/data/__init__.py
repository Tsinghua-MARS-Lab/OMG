__all__ = [
    "G1MotionDataset",
    "GenerationDataModule",
    "motion_collate_fn",
]


def __getattr__(name: str):
    if name in {"GenerationDataModule", "motion_collate_fn"}:
        from omg.data.datamodule import GenerationDataModule, motion_collate_fn

        return {"GenerationDataModule": GenerationDataModule, "motion_collate_fn": motion_collate_fn}[name]
    if name == "G1MotionDataset":
        from omg.data.g1_motion import G1MotionDataset

        return G1MotionDataset
    raise AttributeError(name)
