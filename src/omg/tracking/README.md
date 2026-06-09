# Tracking

This package contains motion tracking integrations. HoloMotion is the supported
downstream tracker for Unitree G1 reference motion.

The integration boundary is `qpos_36` reference motion plus metadata. Tracking
code should depend on shared motion and robot modules, not on generation
internals.

See `docs/tracking.md` for the release workflow.
