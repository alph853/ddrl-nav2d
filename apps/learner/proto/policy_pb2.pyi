from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class PolicyTensor(_message.Message):
    __slots__ = ("name", "shape", "data", "dtype")
    NAME_FIELD_NUMBER: _ClassVar[int]
    SHAPE_FIELD_NUMBER: _ClassVar[int]
    DATA_FIELD_NUMBER: _ClassVar[int]
    DTYPE_FIELD_NUMBER: _ClassVar[int]
    name: str
    shape: _containers.RepeatedScalarFieldContainer[int]
    data: bytes
    dtype: str
    def __init__(self, name: _Optional[str] = ..., shape: _Optional[_Iterable[int]] = ..., data: _Optional[bytes] = ..., dtype: _Optional[str] = ...) -> None: ...

class PolicyPush(_message.Message):
    __slots__ = ("version", "tensors", "scalars", "metadata", "onnx_model")
    class ScalarsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: float
        def __init__(self, key: _Optional[str] = ..., value: _Optional[float] = ...) -> None: ...
    VERSION_FIELD_NUMBER: _ClassVar[int]
    TENSORS_FIELD_NUMBER: _ClassVar[int]
    SCALARS_FIELD_NUMBER: _ClassVar[int]
    METADATA_FIELD_NUMBER: _ClassVar[int]
    ONNX_MODEL_FIELD_NUMBER: _ClassVar[int]
    version: int
    tensors: _containers.RepeatedCompositeFieldContainer[PolicyTensor]
    scalars: _containers.ScalarMap[str, float]
    metadata: bytes
    onnx_model: bytes
    def __init__(self, version: _Optional[int] = ..., tensors: _Optional[_Iterable[_Union[PolicyTensor, _Mapping]]] = ..., scalars: _Optional[_Mapping[str, float]] = ..., metadata: _Optional[bytes] = ..., onnx_model: _Optional[bytes] = ...) -> None: ...

class PolicySubscription(_message.Message):
    __slots__ = ("worker_id", "session_token", "last_version")
    WORKER_ID_FIELD_NUMBER: _ClassVar[int]
    SESSION_TOKEN_FIELD_NUMBER: _ClassVar[int]
    LAST_VERSION_FIELD_NUMBER: _ClassVar[int]
    worker_id: str
    session_token: str
    last_version: int
    def __init__(self, worker_id: _Optional[str] = ..., session_token: _Optional[str] = ..., last_version: _Optional[int] = ...) -> None: ...

class PolicyAck(_message.Message):
    __slots__ = ("worker_id", "version", "received_unix_ms")
    WORKER_ID_FIELD_NUMBER: _ClassVar[int]
    VERSION_FIELD_NUMBER: _ClassVar[int]
    RECEIVED_UNIX_MS_FIELD_NUMBER: _ClassVar[int]
    worker_id: str
    version: int
    received_unix_ms: int
    def __init__(self, worker_id: _Optional[str] = ..., version: _Optional[int] = ..., received_unix_ms: _Optional[int] = ...) -> None: ...

class PolicyStreamRequest(_message.Message):
    __slots__ = ("subscribe", "ack")
    SUBSCRIBE_FIELD_NUMBER: _ClassVar[int]
    ACK_FIELD_NUMBER: _ClassVar[int]
    subscribe: PolicySubscription
    ack: PolicyAck
    def __init__(self, subscribe: _Optional[_Union[PolicySubscription, _Mapping]] = ..., ack: _Optional[_Union[PolicyAck, _Mapping]] = ...) -> None: ...
