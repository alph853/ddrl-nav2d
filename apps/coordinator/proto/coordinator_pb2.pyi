from google.protobuf import empty_pb2 as _empty_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class ControlType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    CONTROL_NONE: _ClassVar[ControlType]
    CONTROL_DRAIN: _ClassVar[ControlType]
    CONTROL_RESUME: _ClassVar[ControlType]
    CONTROL_SWITCH_LEARNER: _ClassVar[ControlType]
    CONTROL_SHUTDOWN: _ClassVar[ControlType]
CONTROL_NONE: ControlType
CONTROL_DRAIN: ControlType
CONTROL_RESUME: ControlType
CONTROL_SWITCH_LEARNER: ControlType
CONTROL_SHUTDOWN: ControlType

class Capability(_message.Message):
    __slots__ = ("name", "value")
    NAME_FIELD_NUMBER: _ClassVar[int]
    VALUE_FIELD_NUMBER: _ClassVar[int]
    name: str
    value: str
    def __init__(self, name: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...

class WorkerHello(_message.Message):
    __slots__ = ("hostname", "version", "capabilities", "labels")
    class LabelsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    HOSTNAME_FIELD_NUMBER: _ClassVar[int]
    VERSION_FIELD_NUMBER: _ClassVar[int]
    CAPABILITIES_FIELD_NUMBER: _ClassVar[int]
    LABELS_FIELD_NUMBER: _ClassVar[int]
    hostname: str
    version: str
    capabilities: _containers.RepeatedCompositeFieldContainer[Capability]
    labels: _containers.ScalarMap[str, str]
    def __init__(self, hostname: _Optional[str] = ..., version: _Optional[str] = ..., capabilities: _Optional[_Iterable[_Union[Capability, _Mapping]]] = ..., labels: _Optional[_Mapping[str, str]] = ...) -> None: ...

class WorkerSession(_message.Message):
    __slots__ = ("worker_id", "session_token", "learner_target", "heartbeat_interval_ms")
    WORKER_ID_FIELD_NUMBER: _ClassVar[int]
    SESSION_TOKEN_FIELD_NUMBER: _ClassVar[int]
    LEARNER_TARGET_FIELD_NUMBER: _ClassVar[int]
    HEARTBEAT_INTERVAL_MS_FIELD_NUMBER: _ClassVar[int]
    worker_id: str
    session_token: str
    learner_target: str
    heartbeat_interval_ms: int
    def __init__(self, worker_id: _Optional[str] = ..., session_token: _Optional[str] = ..., learner_target: _Optional[str] = ..., heartbeat_interval_ms: _Optional[int] = ...) -> None: ...

class WorkerStatus(_message.Message):
    __slots__ = ("phase", "active_simulators", "detail")
    PHASE_FIELD_NUMBER: _ClassVar[int]
    ACTIVE_SIMULATORS_FIELD_NUMBER: _ClassVar[int]
    DETAIL_FIELD_NUMBER: _ClassVar[int]
    phase: str
    active_simulators: int
    detail: str
    def __init__(self, phase: _Optional[str] = ..., active_simulators: _Optional[int] = ..., detail: _Optional[str] = ...) -> None: ...

class WorkerHeartbeat(_message.Message):
    __slots__ = ("worker_id", "session_token", "unix_ms", "status", "metrics")
    class MetricsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: float
        def __init__(self, key: _Optional[str] = ..., value: _Optional[float] = ...) -> None: ...
    WORKER_ID_FIELD_NUMBER: _ClassVar[int]
    SESSION_TOKEN_FIELD_NUMBER: _ClassVar[int]
    UNIX_MS_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    METRICS_FIELD_NUMBER: _ClassVar[int]
    worker_id: str
    session_token: str
    unix_ms: int
    status: WorkerStatus
    metrics: _containers.ScalarMap[str, float]
    def __init__(self, worker_id: _Optional[str] = ..., session_token: _Optional[str] = ..., unix_ms: _Optional[int] = ..., status: _Optional[_Union[WorkerStatus, _Mapping]] = ..., metrics: _Optional[_Mapping[str, float]] = ...) -> None: ...

class ControlDirective(_message.Message):
    __slots__ = ("type", "message", "learner_target")
    TYPE_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    LEARNER_TARGET_FIELD_NUMBER: _ClassVar[int]
    type: ControlType
    message: str
    learner_target: str
    def __init__(self, type: _Optional[_Union[ControlType, str]] = ..., message: _Optional[str] = ..., learner_target: _Optional[str] = ...) -> None: ...

class LearnerHello(_message.Message):
    __slots__ = ("endpoint", "version", "models", "hostname")
    ENDPOINT_FIELD_NUMBER: _ClassVar[int]
    VERSION_FIELD_NUMBER: _ClassVar[int]
    MODELS_FIELD_NUMBER: _ClassVar[int]
    HOSTNAME_FIELD_NUMBER: _ClassVar[int]
    endpoint: str
    version: str
    models: _containers.RepeatedScalarFieldContainer[str]
    hostname: str
    def __init__(self, endpoint: _Optional[str] = ..., version: _Optional[str] = ..., models: _Optional[_Iterable[str]] = ..., hostname: _Optional[str] = ...) -> None: ...

class LearnerSession(_message.Message):
    __slots__ = ("learner_id", "session_token", "heartbeat_interval_ms")
    LEARNER_ID_FIELD_NUMBER: _ClassVar[int]
    SESSION_TOKEN_FIELD_NUMBER: _ClassVar[int]
    HEARTBEAT_INTERVAL_MS_FIELD_NUMBER: _ClassVar[int]
    learner_id: str
    session_token: str
    heartbeat_interval_ms: int
    def __init__(self, learner_id: _Optional[str] = ..., session_token: _Optional[str] = ..., heartbeat_interval_ms: _Optional[int] = ...) -> None: ...

class LearnerHeartbeat(_message.Message):
    __slots__ = ("learner_id", "session_token", "unix_ms", "endpoint", "version", "models", "metrics")
    class MetricsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: float
        def __init__(self, key: _Optional[str] = ..., value: _Optional[float] = ...) -> None: ...
    LEARNER_ID_FIELD_NUMBER: _ClassVar[int]
    SESSION_TOKEN_FIELD_NUMBER: _ClassVar[int]
    UNIX_MS_FIELD_NUMBER: _ClassVar[int]
    ENDPOINT_FIELD_NUMBER: _ClassVar[int]
    VERSION_FIELD_NUMBER: _ClassVar[int]
    MODELS_FIELD_NUMBER: _ClassVar[int]
    METRICS_FIELD_NUMBER: _ClassVar[int]
    learner_id: str
    session_token: str
    unix_ms: int
    endpoint: str
    version: str
    models: _containers.RepeatedScalarFieldContainer[str]
    metrics: _containers.ScalarMap[str, float]
    def __init__(self, learner_id: _Optional[str] = ..., session_token: _Optional[str] = ..., unix_ms: _Optional[int] = ..., endpoint: _Optional[str] = ..., version: _Optional[str] = ..., models: _Optional[_Iterable[str]] = ..., metrics: _Optional[_Mapping[str, float]] = ...) -> None: ...

class LearnerHeartbeatRequest(_message.Message):
    __slots__ = ("heartbeat",)
    HEARTBEAT_FIELD_NUMBER: _ClassVar[int]
    heartbeat: LearnerHeartbeat
    def __init__(self, heartbeat: _Optional[_Union[LearnerHeartbeat, _Mapping]] = ...) -> None: ...
