from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "ExecutedHistoryBuffer": ("omg.realtime.motion_buffer", "ExecutedHistoryBuffer"),
    "PlanSegment": ("omg.realtime.motion_buffer", "PlanSegment"),
    "ReferenceMotionBuffer": ("omg.realtime.motion_buffer", "ReferenceMotionBuffer"),
    "RealtimeOrinBufferClient": ("omg.realtime.orin_client", "RealtimeOrinBufferClient"),
    "RealtimeOrinBufferClientConfig": ("omg.realtime.orin_client", "RealtimeOrinBufferClientConfig"),
    "MotionPlanChunk": ("omg.realtime.protocol", "MotionPlanChunk"),
    "RobotStateRequest": ("omg.realtime.protocol", "RobotStateRequest"),
    "ZmqPlanClient": ("omg.realtime.transport", "ZmqPlanClient"),
    "ZmqPlanServer": ("omg.realtime.transport", "ZmqPlanServer"),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    try:
        module_name, attr_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
