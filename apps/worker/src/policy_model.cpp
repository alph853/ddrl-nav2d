#include "worker/policy_model.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <string_view>

#include "worker/observation_encoder.hpp"
#include "worker/policy_subscriber.hpp"

namespace ddrl::worker {

namespace {

constexpr std::string_view kInputObservation{"observation"};
constexpr std::string_view kInputHiddenIn{"hidden_in"};
constexpr std::string_view kInputCellIn{"cell_in"};
constexpr std::string_view kOutputMean{"mean"};
constexpr std::string_view kOutputLogStd{"log_std"};
constexpr std::string_view kOutputHiddenOut{"hidden_out"};
constexpr std::string_view kOutputCellOut{"cell_out"};

[[nodiscard]] Ort::SessionOptions make_session_options()
{
  Ort::SessionOptions opts = PolicyModel::default_session_options();
  opts.SetIntraOpNumThreads(1);
  opts.SetInterOpNumThreads(1);
  opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
  return opts;
}

} // namespace

Ort::Env& PolicyModel::ort_env()
{
  static Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "ddrl_worker_policy"};
  return env;
}

Ort::SessionOptions PolicyModel::default_session_options()
{
  Ort::SessionOptions opts;
  opts.SetIntraOpNumThreads(1);
  opts.SetInterOpNumThreads(1);
  return opts;
}

PolicyModel::PolicyModel(
  config::RLPolicyConfig::Architecture arch, std::shared_ptr<logging::Logger> logger
)
    : arch_(arch),
      logger_(std::move(logger)),
      memory_info_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)),
      observation_dim_(arch.observation_dim())
{
}

std::shared_ptr<PolicyModel> PolicyModel::from_payload(
  const PolicyPayload& payload, const config::RLPolicyConfig& policy_config,
  const std::shared_ptr<logging::Logger>& logger
)
{
  auto model = std::shared_ptr<PolicyModel>(new PolicyModel(policy_config.architecture, logger));
  model->version_ = payload.version;

  if (!model->load_onnx(payload)) {
    return nullptr;
  }
  return model;
}

bool PolicyModel::load_onnx(const PolicyPayload& payload)
{
  if (payload.onnx_model.empty()) {
    if (logger_) {
      DDRL_LOG_ERROR(logger_, "Policy payload v{} is missing ONNX model bytes", payload.version);
    }
    return false;
  }

  auto         opts   = make_session_options();
  auto&        env    = ort_env();
  const void*  buffer = payload.onnx_model.data();
  const size_t size   = payload.onnx_model.size();

  try {
    session_ = std::make_unique<Ort::Session>(env, buffer, size, opts);
  } catch (const Ort::Exception& e) {
    if (logger_) {
      DDRL_LOG_ERROR(logger_, "Failed to load ONNX model v{}: {}", payload.version, e.what());
    }
    session_.reset();
    return false;
  }

  // Infer output dimensions from model outputs if possible.
  try {
    auto type_info   = session_->GetOutputTypeInfo(0);
    auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
    auto shape       = tensor_info.GetShape();
    if (shape.size() >= 3) {
      action_dim_ = static_cast<size_t>(std::max<int64_t>(1, shape.back()));
    }

  } catch (const Ort::Exception& e) {
    if (logger_) {
      DDRL_LOG_WARN(logger_, "Could not infer output dimensions from ONNX: {}", e.what());
    }
  }

  try {
    const auto input_count = session_->GetInputCount();
    recurrent_             = input_count >= 3;
    if (recurrent_) {
      auto hidden_type_info   = session_->GetInputTypeInfo(1);
      auto hidden_tensor_info = hidden_type_info.GetTensorTypeAndShapeInfo();
      auto hidden_shape       = hidden_tensor_info.GetShape();
      if (hidden_shape.size() >= 3) {
        recurrent_hidden_dim_ = static_cast<size_t>(std::max<int64_t>(1, hidden_shape.back()));
      }
    }
  } catch (const Ort::Exception& e) {
    if (logger_) {
      DDRL_LOG_WARN(logger_, "Could not infer recurrent state dimensions from ONNX: {}", e.what());
    }
    recurrent_          = false;
    recurrent_hidden_dim_ = 0;
  }

  if (action_dim_ == 0) {
    action_dim_ = std::max<size_t>(1, static_cast<size_t>(arch_.action_dim));
  }
  return true;
}

void PolicyModel::allocate_context(PolicyContext& ctx) const
{
  if (recurrent_) {
    if (ctx.hidden_state.size() != recurrent_hidden_dim_) {
      ctx.hidden_state.assign(recurrent_hidden_dim_, 0.0f);
    }
    if (ctx.cell_state.size() != recurrent_hidden_dim_) {
      ctx.cell_state.assign(recurrent_hidden_dim_, 0.0f);
    }
  }
}

bool PolicyModel::evaluate(
  const core::rl::Observation& obs, PolicyContext& context, std::vector<float>& mean_out,
  std::vector<float>& log_std_out
) const
{
  auto enc = encode_observation(obs, arch_);
  return evaluate_encoded(enc.features, context, mean_out, log_std_out);
}

bool PolicyModel::evaluate_encoded(
  std::span<const float> encoded_observation, PolicyContext& context, std::vector<float>& mean_out,
  std::vector<float>& log_std_out
) const
{
  if (!session_) {
    if (logger_) {
      DDRL_LOG_ERROR(logger_, "PolicyModel evaluate called without loaded ONNX session");
    }
    return false;
  }

  allocate_context(context);

  std::vector<float> observation(observation_dim_, 0.0f);
  const auto copy_dim = std::min<std::size_t>(observation_dim_, encoded_observation.size());
  if (copy_dim > 0) {
    std::copy_n(encoded_observation.data(), copy_dim, observation.data());
  }
  if (std::any_of(observation.begin(), observation.end(), [](float v) { return !std::isfinite(v); })) {
    if (logger_) {
      DDRL_LOG_WARN(logger_, "encoded observation contains non-finite values; refusing policy eval");
    }
    return false;
  }

  const std::array<int64_t, 3> observation_shape{1, 1, static_cast<int64_t>(observation_dim_)};
  const std::array<int64_t, 3> recurrent_shape{1, 1, static_cast<int64_t>(recurrent_hidden_dim_)};

  std::vector<Ort::Value> inputs;
  inputs.reserve(recurrent_ ? 3 : 1);
  inputs.emplace_back(Ort::Value::CreateTensor<float>(
    memory_info_,
    observation.data(),
    observation.size(),
    observation_shape.data(),
    observation_shape.size()
  ));
  if (recurrent_) {
    inputs.emplace_back(Ort::Value::CreateTensor<float>(
      memory_info_,
      context.hidden_state.data(),
      context.hidden_state.size(),
      recurrent_shape.data(),
      recurrent_shape.size()
    ));
    inputs.emplace_back(Ort::Value::CreateTensor<float>(
      memory_info_,
      context.cell_state.data(),
      context.cell_state.size(),
      recurrent_shape.data(),
      recurrent_shape.size()
    ));
  }
  const std::array<const char*, 3> input_names{
    kInputObservation.data(),
    kInputHiddenIn.data(),
    kInputCellIn.data(),
  };
  const std::array<const char*, 4> output_names{
    kOutputMean.data(),
    kOutputLogStd.data(),
    kOutputHiddenOut.data(),
    kOutputCellOut.data(),
  };

  std::vector<Ort::Value> outputs;
  try {
    outputs = session_->Run(
      Ort::RunOptions{nullptr},
      input_names.data(),
      inputs.data(),
      inputs.size(),
      output_names.data(),
      recurrent_ ? 4 : 2
    );
  } catch (const Ort::Exception& e) {
    if (logger_) {
      DDRL_LOG_ERROR(logger_, "ONNX inference failed: {}", e.what());
    }
    return false;
  }

  if (outputs.size() < 2) {
    if (logger_) {
      DDRL_LOG_ERROR(logger_, "ONNX inference returned {} outputs, expected 2", outputs.size());
    }
    return false;
  }

  auto* mean_tensor    = outputs[0].GetTensorMutableData<float>();
  auto* log_std_tensor = outputs[1].GetTensorMutableData<float>();

  mean_out.assign(mean_tensor, mean_tensor + action_dim_);
  log_std_out.assign(log_std_tensor, log_std_tensor + action_dim_);
  if (recurrent_ && outputs.size() >= 4) {
    auto* hidden_out_tensor = outputs[2].GetTensorMutableData<float>();
    auto* cell_out_tensor   = outputs[3].GetTensorMutableData<float>();
    context.hidden_state.assign(hidden_out_tensor, hidden_out_tensor + recurrent_hidden_dim_);
    context.cell_state.assign(cell_out_tensor, cell_out_tensor + recurrent_hidden_dim_);
  }
  const bool outputs_finite =
    std::all_of(mean_out.begin(), mean_out.end(), [](float v) { return std::isfinite(v); }) &&
    std::all_of(log_std_out.begin(), log_std_out.end(), [](float v) { return std::isfinite(v); });
  if (!outputs_finite) {
    if (logger_) {
      DDRL_LOG_WARN(logger_, "policy model produced non-finite outputs; refusing policy eval");
    }
    return false;
  }

  return true;
}

} // namespace ddrl::worker
