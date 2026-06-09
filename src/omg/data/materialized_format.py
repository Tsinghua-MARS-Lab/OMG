from __future__ import annotations

TENSOR_KEYS = (
    "length",
    "fps",
    "qpos_36",
    "body_pos_w",
    "body_quat_w",
    "audio_features",
    "human_motion",
    "motion_features",
    "root_pos_local",
    "root_rot_local_quat",
    "joint_dof",
    "body_link_pos_local",
    "prev_state_features",
    "history_features",
    "canonical_frame_idx",
    "canon_root_pos",
    "canon_root_quat",
    "has_text",
)

MASK_KEYS = ("valid", "has_audio", "has_human_motion")

DEFAULT_TRAIN_TENSOR_KEYS = (
    "fps",
    "audio_features",
    "human_motion",
    "motion_features",
    "history_features",
    "canon_root_pos",
    "canon_root_quat",
    "has_text",
)
