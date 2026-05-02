from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class ActorId(_message.Message):
    __slots__ = ("worker_id", "sim_id", "rollout_id")
    WORKER_ID_FIELD_NUMBER: _ClassVar[int]
    SIM_ID_FIELD_NUMBER: _ClassVar[int]
    ROLLOUT_ID_FIELD_NUMBER: _ClassVar[int]
    worker_id: int
    sim_id: int
    rollout_id: int
    def __init__(self, worker_id: _Optional[int] = ..., sim_id: _Optional[int] = ..., rollout_id: _Optional[int] = ...) -> None: ...

class Observation(_message.Message):
    __slots__ = ("features",)
    FEATURES_FIELD_NUMBER: _ClassVar[int]
    features: _containers.RepeatedScalarFieldContainer[float]
    def __init__(self, features: _Optional[_Iterable[float]] = ...) -> None: ...

class Action(_message.Message):
    __slots__ = ("continuous", "log_prob")
    CONTINUOUS_FIELD_NUMBER: _ClassVar[int]
    LOG_PROB_FIELD_NUMBER: _ClassVar[int]
    continuous: _containers.RepeatedScalarFieldContainer[float]
    log_prob: float
    def __init__(self, continuous: _Optional[_Iterable[float]] = ..., log_prob: _Optional[float] = ...) -> None: ...

class RewardTermValue(_message.Message):
    __slots__ = ("raw", "weighted")
    RAW_FIELD_NUMBER: _ClassVar[int]
    WEIGHTED_FIELD_NUMBER: _ClassVar[int]
    raw: float
    weighted: float
    def __init__(self, raw: _Optional[float] = ..., weighted: _Optional[float] = ...) -> None: ...

class RecurrentState(_message.Message):
    __slots__ = ("hidden", "cell", "shape")
    HIDDEN_FIELD_NUMBER: _ClassVar[int]
    CELL_FIELD_NUMBER: _ClassVar[int]
    SHAPE_FIELD_NUMBER: _ClassVar[int]
    hidden: _containers.RepeatedScalarFieldContainer[float]
    cell: _containers.RepeatedScalarFieldContainer[float]
    shape: _containers.RepeatedScalarFieldContainer[int]
    def __init__(self, hidden: _Optional[_Iterable[float]] = ..., cell: _Optional[_Iterable[float]] = ..., shape: _Optional[_Iterable[int]] = ...) -> None: ...

class TrajectoryStep(_message.Message):
    __slots__ = ("observation", "action", "reward", "terminal", "timestamp_sec", "raw_action", "reward_terms")
    class RewardTermsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: RewardTermValue
        def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[RewardTermValue, _Mapping]] = ...) -> None: ...
    OBSERVATION_FIELD_NUMBER: _ClassVar[int]
    ACTION_FIELD_NUMBER: _ClassVar[int]
    REWARD_FIELD_NUMBER: _ClassVar[int]
    TERMINAL_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_SEC_FIELD_NUMBER: _ClassVar[int]
    RAW_ACTION_FIELD_NUMBER: _ClassVar[int]
    REWARD_TERMS_FIELD_NUMBER: _ClassVar[int]
    observation: Observation
    action: Action
    reward: float
    terminal: bool
    timestamp_sec: float
    raw_action: _containers.RepeatedScalarFieldContainer[float]
    reward_terms: _containers.MessageMap[str, RewardTermValue]
    def __init__(self, observation: _Optional[_Union[Observation, _Mapping]] = ..., action: _Optional[_Union[Action, _Mapping]] = ..., reward: _Optional[float] = ..., terminal: bool = ..., timestamp_sec: _Optional[float] = ..., raw_action: _Optional[_Iterable[float]] = ..., reward_terms: _Optional[_Mapping[str, RewardTermValue]] = ...) -> None: ...

class Rollout(_message.Message):
    __slots__ = ("actor", "policy_version", "episode_return", "episode_terminal", "steps", "metrics", "initial_recurrent_state")
    class MetricsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: float
        def __init__(self, key: _Optional[str] = ..., value: _Optional[float] = ...) -> None: ...
    ACTOR_FIELD_NUMBER: _ClassVar[int]
    POLICY_VERSION_FIELD_NUMBER: _ClassVar[int]
    EPISODE_RETURN_FIELD_NUMBER: _ClassVar[int]
    EPISODE_TERMINAL_FIELD_NUMBER: _ClassVar[int]
    STEPS_FIELD_NUMBER: _ClassVar[int]
    METRICS_FIELD_NUMBER: _ClassVar[int]
    INITIAL_RECURRENT_STATE_FIELD_NUMBER: _ClassVar[int]
    actor: ActorId
    policy_version: int
    episode_return: float
    episode_terminal: bool
    steps: _containers.RepeatedCompositeFieldContainer[TrajectoryStep]
    metrics: _containers.ScalarMap[str, float]
    initial_recurrent_state: RecurrentState
    def __init__(self, actor: _Optional[_Union[ActorId, _Mapping]] = ..., policy_version: _Optional[int] = ..., episode_return: _Optional[float] = ..., episode_terminal: bool = ..., steps: _Optional[_Iterable[_Union[TrajectoryStep, _Mapping]]] = ..., metrics: _Optional[_Mapping[str, float]] = ..., initial_recurrent_state: _Optional[_Union[RecurrentState, _Mapping]] = ...) -> None: ...

class RolloutBatch(_message.Message):
    __slots__ = ("rollouts", "batch_id", "created_unix_ms")
    ROLLOUTS_FIELD_NUMBER: _ClassVar[int]
    BATCH_ID_FIELD_NUMBER: _ClassVar[int]
    CREATED_UNIX_MS_FIELD_NUMBER: _ClassVar[int]
    rollouts: _containers.RepeatedCompositeFieldContainer[Rollout]
    batch_id: int
    created_unix_ms: int
    def __init__(self, rollouts: _Optional[_Iterable[_Union[Rollout, _Mapping]]] = ..., batch_id: _Optional[int] = ..., created_unix_ms: _Optional[int] = ...) -> None: ...

class RolloutStreamRequest(_message.Message):
    __slots__ = ("worker_id", "session_token", "batch")
    WORKER_ID_FIELD_NUMBER: _ClassVar[int]
    SESSION_TOKEN_FIELD_NUMBER: _ClassVar[int]
    BATCH_FIELD_NUMBER: _ClassVar[int]
    worker_id: str
    session_token: str
    batch: RolloutBatch
    def __init__(self, worker_id: _Optional[str] = ..., session_token: _Optional[str] = ..., batch: _Optional[_Union[RolloutBatch, _Mapping]] = ...) -> None: ...

class RolloutStreamResponse(_message.Message):
    __slots__ = ("last_batch_id", "message")
    LAST_BATCH_ID_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    last_batch_id: int
    message: str
    def __init__(self, last_batch_id: _Optional[int] = ..., message: _Optional[str] = ...) -> None: ...
