from __future__ import annotations

import time
from typing import Any

from omg.realtime.protocol import (
    MotionPlanChunk,
    RobotStateRequest,
    decode_message,
)


def _zmq() -> Any:
    try:
        import zmq
    except ImportError as exc:  # pragma: no cover - optional dependency boundary
        raise RuntimeError("Realtime planning transport requires pyzmq. Install omg[realtime].") from exc
    return zmq


def recv_realtime_message(socket: Any) -> tuple[dict, dict]:
    frames = socket.recv_multipart()
    if len(frames) != 2:
        raise RuntimeError(f"Realtime protocol expects 2 multipart frames, got {len(frames)}")
    return decode_message(frames[0], frames[1])


def send_realtime_message(socket: Any, message: tuple[bytes, bytes]) -> None:
    socket.send_multipart([message[0], message[1]])


class ZmqPlanServer:
    def __init__(self, bind: str, *, linger_ms: int = 0) -> None:
        zmq = _zmq()
        self._ctx = zmq.Context.instance()
        self._socket = self._ctx.socket(zmq.REP)
        self._socket.setsockopt(zmq.LINGER, int(linger_ms))
        self._socket.bind(str(bind))

    def recv_request(self) -> RobotStateRequest:
        header, arrays = recv_realtime_message(self._socket)
        request = RobotStateRequest.from_message(header, arrays)
        request.metadata.setdefault("realtime_transport", {})
        request.metadata["realtime_transport"]["server_recv_wall_time"] = time.time()
        request.metadata["realtime_transport"]["server_recv_perf_time"] = time.perf_counter()
        return request

    def send_plan(self, plan: MotionPlanChunk) -> None:
        plan.metadata.setdefault("realtime_transport", {})
        plan.metadata["realtime_transport"]["server_send_wall_time"] = time.time()
        plan.metadata["realtime_transport"]["server_send_perf_time"] = time.perf_counter()
        send_realtime_message(self._socket, plan.to_message())

    def close(self) -> None:
        self._socket.close()

    def __enter__(self) -> "ZmqPlanServer":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()


class ZmqPlanClient:
    def __init__(self, connect: str, *, linger_ms: int = 0) -> None:
        zmq = _zmq()
        self._zmq = zmq
        self._ctx = zmq.Context.instance()
        self._socket = self._ctx.socket(zmq.REQ)
        self._socket.setsockopt(zmq.LINGER, int(linger_ms))
        self._socket.connect(str(connect))
        self._pending_request_id: str | None = None
        self._client_send_start_wall_time: float | None = None
        self._client_send_start_perf_time: float | None = None
        self._client_send_end_perf_time: float | None = None

    def request_plan(self, request: RobotStateRequest, *, timeout_ms: int = 30000) -> MotionPlanChunk:
        timeout = int(timeout_ms)
        if timeout <= 0:
            raise ValueError(f"timeout_ms must be positive, got {timeout_ms}")
        self.begin_request(request)
        plan = self.poll_plan(timeout_ms=timeout)
        if plan is None:
            raise TimeoutError(f"Timed out waiting {timeout}ms for realtime plan response")
        return plan

    def begin_request(self, request: RobotStateRequest) -> None:
        if self._pending_request_id is not None:
            raise RuntimeError(f"Realtime plan request already pending: {self._pending_request_id}")
        self._client_send_start_wall_time = time.time()
        self._client_send_start_perf_time = time.perf_counter()
        send_realtime_message(self._socket, request.to_message())
        self._client_send_end_perf_time = time.perf_counter()
        self._pending_request_id = request.request_id

    def poll_plan(self, *, timeout_ms: int = 0) -> MotionPlanChunk | None:
        if self._pending_request_id is None:
            raise RuntimeError("No realtime plan request is pending")
        timeout = int(timeout_ms)
        if timeout < 0:
            raise ValueError(f"timeout_ms must be non-negative, got {timeout_ms}")
        events = int(self._socket.poll(timeout))
        client_poll_end_perf_time = time.perf_counter()
        if (events & self._zmq.POLLIN) == 0:
            return None
        header, arrays = recv_realtime_message(self._socket)
        client_recv_end_wall_time = time.time()
        client_recv_end_perf_time = time.perf_counter()
        plan = MotionPlanChunk.from_message(header, arrays)
        pending_request_id = self._pending_request_id
        client_send_start_wall_time = float(self._client_send_start_wall_time)
        client_send_start_perf_time = float(self._client_send_start_perf_time)
        client_send_end_perf_time = float(self._client_send_end_perf_time)
        self._pending_request_id = None
        self._client_send_start_wall_time = None
        self._client_send_start_perf_time = None
        self._client_send_end_perf_time = None
        if plan.request_id != pending_request_id:
            raise RuntimeError(f"Plan response request_id mismatch: {plan.request_id} != {pending_request_id}")
        transport = plan.metadata.setdefault("realtime_transport", {})
        client_send_ms = (client_send_end_perf_time - client_send_start_perf_time) * 1000.0
        client_wait_ms = (client_poll_end_perf_time - client_send_end_perf_time) * 1000.0
        client_recv_decode_ms = (client_recv_end_perf_time - client_poll_end_perf_time) * 1000.0
        client_total_ms = (client_recv_end_perf_time - client_send_start_perf_time) * 1000.0
        server_plan_ms = float(plan.planning_latency_seconds) * 1000.0
        transport.update(
            {
                "client_send_start_wall_time": client_send_start_wall_time,
                "client_recv_end_wall_time": client_recv_end_wall_time,
                "client_send_ms": client_send_ms,
                "client_wait_for_reply_ms": client_wait_ms,
                "client_recv_decode_ms": client_recv_decode_ms,
                "client_request_total_ms": client_total_ms,
                "client_network_queue_total_estimate_ms": max(
                    client_total_ms - client_send_ms - client_recv_decode_ms - server_plan_ms,
                    0.0,
                ),
            }
        )
        server_recv_wall = transport.get("server_recv_wall_time")
        server_send_wall = transport.get("server_send_wall_time")
        if server_recv_wall is not None and server_send_wall is not None:
            transport["wall_clock_skew_estimate_ms"] = (
                (float(server_recv_wall) + float(server_send_wall)) * 0.5
                - (client_send_start_wall_time + client_recv_end_wall_time) * 0.5
            ) * 1000.0
        return plan

    def close(self) -> None:
        self._socket.close()

    def __enter__(self) -> "ZmqPlanClient":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()
