/**
 * @file event_bus.hpp
 * @brief Lock-free event bus using single-producer single-consumer queues
 */

#pragma once

#include "core/base/result.hpp"
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <readerwriterqueue.h>
#include <shared_mutex>
#include <string_view>
#include <unordered_map>
#include <variant>

namespace ddrl::events {

// namespace qos {
// enum class Reliability : uint8_t { BEST_EFFORT, RELIABLE };
// // enum class Durability : uint8_t { VOLATILE, TRANSIENT_LOCAL };
// } // namespace qos

// struct QoS {
//   qos::Reliability reliability = qos::Reliability::BEST_EFFORT;
//   // qos::Durability  durability  = qos::Durability::VOLATILE;
//   size_t queue_capacity = 1024;
//   size_t keep_last      = 10;
// };

/**
 * @brief Lock-free event bus using single-producer single-consumer queues
 * @note register_queue(...) must be called before publishing or subscribing
 *
 * Each (type, entity_id) pair has a dedicated SPSC queue
 * (moodycamel::ReaderWriterQueue) publish(...) enqueues directly to the
 * target entity's queue (no blocking callbacks) Consumers poll their dedicated
 * queues via try_dequeue(...) No runtime type checks, no any_cast, no mutex
 * contention in hot path
 */
class EventBus
{
public:
  using EntityId = uint32_t;

  EventBus()  = default;
  ~EventBus() = default;

  // Non-copyable, non-movable
  EventBus(const EventBus&)            = delete;
  EventBus& operator=(const EventBus&) = delete;
  EventBus(EventBus&&)                 = delete;
  EventBus& operator=(EventBus&&)      = delete;

  // ======================================================
  // --------------  Event Registration API ---------------
  // ======================================================

  template <class T>
  core::Result<void> register_queue(EntityId id, size_t capacity = 1024, size_t keep_last = 0);

  template <class T>
  core::Result<void> unregister_queue(EntityId id);

  template <class T>
  [[nodiscard]] bool is_registered(EntityId id) const;

  // ======================================================
  // --------------  Producer/Consumer APIs ---------------
  // ======================================================

  template <typename T>
  core::Result<void> publish(EntityId entity_id, T&& msg);

  template <typename T>
  core::Result<void> try_dequeue(EntityId entity_id, T& msg);

  template <class T>
  core::Result<T> try_dequeue(EntityId id);

  template <class T>
  void clear();

private:
  /**
   * @brief Key for (entity_id, type) routing in queue map
   */
  struct Key {
    EntityId id;

    bool operator==(const Key& other) const { return id == other.id; }
  };

  /**
   * @brief Hash function for Key
   */
  struct KeyHash {
    size_t operator()(const Key& k) const noexcept { return std::hash<EntityId>{}(k.id); }
  };

  template <class T>
  class Channel
  {
  public:
    using QType = moodycamel::ReaderWriterQueue<T>; // SPSC
    using Mailbox = std::atomic<std::shared_ptr<T>>; // Single-slot mailbox for keep_last=1

    core::Result<void> register_queue(EntityId id, size_t cap, size_t keep_last = 0)
    {
      std::unique_lock lk(mx_);
      using namespace core;

      Key key{id};
      if (map_.find(key) != map_.end()) {
        return already_registered_err(make_context("Channel<T>::register_queue", "entity_id", id));
      }

      // Use mailbox if keep_last=1, otherwise use queue
      if (keep_last == 1) {
        auto mailbox = std::make_shared<Mailbox>(nullptr);
        map_.emplace(key, Entry{std::move(mailbox), keep_last});
      } else {
        auto q = std::make_shared<QType>(cap);
        map_.emplace(key, Entry{std::move(q), keep_last});
      }
      return {};
    }

    core::Result<void> unregister_queue(EntityId id)
    {
      std::unique_lock lk(mx_);
      using namespace core;

      Key  key{id};
      auto it = map_.find(key);
      if (it == map_.end()) {
        return not_found_err(make_context("Channel<T>::unregister_queue", "entity_id", id));
      }
      map_.erase(it);
      return {};
    }

    bool is_registered(EntityId id) const
    {
      std::shared_lock lk(mx_);
      return map_.find(Key{id}) != map_.end();
    }

    core::Result<void> publish(EntityId id, const T& v)
    {
      using namespace core;
      auto entry_opt = get_entry(id);

      auto ctx = [&] { return make_context("Channel<T>::publish", "entity_id", id); };
      if (!entry_opt) {
        return not_found_err(ctx());
      }

      const auto& entry = *entry_opt;

      // Handle mailbox mode (keep_last=1)
      if (entry.keep_last == 1) {
        auto mailbox_ptr = std::get<std::shared_ptr<Mailbox>>(entry.storage);
        mailbox_ptr->store(std::make_shared<T>(v), std::memory_order_release);
        return {};
      }

      // Handle queue mode
      auto q = std::get<std::shared_ptr<QType>>(entry.storage);

      // Try to enqueue
      if (!q->try_enqueue(v)) {
        // If queue is full and keep_last is enabled, make space by dequeueing old messages
        if (entry.keep_last > 0) {
          T discard;
          q->try_dequeue(discard); // Remove oldest message
          if (!q->try_enqueue(v)) {
            return queue_overflow_err(ctx()); // Still failed, return error
          }
        } else {
          return queue_overflow_err(ctx());
        }
      }
      return {};
    }
    core::Result<void> publish(EntityId id, T&& v)
    {
      using namespace core;
      auto entry_opt = get_entry(id);

      auto ctx = [&] { return make_context("Channel<T>::publish", "entity_id", id); };

      if (!entry_opt) {
        return not_found_err(ctx());
      }

      const auto& entry = *entry_opt;

      // Handle mailbox mode (keep_last=1)
      if (entry.keep_last == 1) {
        auto mailbox_ptr = std::get<std::shared_ptr<Mailbox>>(entry.storage);
        mailbox_ptr->store(std::make_shared<T>(std::move(v)), std::memory_order_release);
        return {};
      }

      // Handle queue mode
      auto q = std::get<std::shared_ptr<QType>>(entry.storage);

      // If keep_last is enabled, pre-emptively make space
      if (entry.keep_last > 0) {
        T discard;
        q->try_dequeue(discard); // Remove oldest message to make room
      }

      // Now try to enqueue (only move once!)
      if (!q->try_enqueue(std::move(v))) {
        return queue_overflow_err(ctx());
      }
      return {};
    }

    core::Result<T> try_dequeue(EntityId id)
    {
      using namespace core;
      auto entry_opt = get_entry(id);

      auto ctx = [&] { return make_context("Channel<T>::try_dequeue", "entity_id", id); };

      if (!entry_opt) {
        return std::unexpected(not_found_err(ctx()).error());
      }

      const auto& entry = *entry_opt;

      // Handle mailbox mode (keep_last=1)
      if (entry.keep_last == 1) {
        auto mailbox_ptr = std::get<std::shared_ptr<Mailbox>>(entry.storage);
        auto msg_ptr = mailbox_ptr->exchange(nullptr, std::memory_order_acquire);
        if (!msg_ptr) {
          return std::unexpected(no_message_available_err(ctx()));
        }
        return std::move(*msg_ptr);
      }

      // Handle queue mode
      auto q = std::get<std::shared_ptr<QType>>(entry.storage);
      T out;
      if (!q->try_dequeue(out)) {
        return std::unexpected(no_message_available_err(ctx()));
      }
      return out;
    }

    void clear()
    {
      std::unique_lock lk(mx_);
      map_.clear();
    }

  private:
    core::Result<void> not_found_err(std::string_view ctx) const
    {
      return std::unexpected(
        core::make_error(core::ErrorCode::EVENT_BUS, "Queue not registered to the event bus.", ctx)
      );
    }

    core::Result<void> queue_overflow_err(std::string_view ctx) const
    {
      return std::unexpected(core::make_error(
        core::ErrorCode::EVENT_BUS, "Failed to publish/enqueue message: queue overflow", ctx
      ));
    }

    core::Result<void> already_registered_err(std::string_view ctx) const
    {
      return std::unexpected(core::make_error(
        core::ErrorCode::EVENT_BUS_REGISTERED, "Queue already registered to the event bus.", ctx
      ));
    }

    core::Error no_message_available_err(std::string_view ctx) const
    {
      return core::make_error(core::ErrorCode::EVENT_BUS, "No message available", ctx);
    }

    struct Entry {
      std::variant<std::shared_ptr<QType>, std::shared_ptr<Mailbox>> storage;
      size_t keep_last = 0; // 0 means no limit, 1 means mailbox mode, >1 means keep only last N messages
    };

    std::optional<Entry> get_entry(EntityId id) const
    {
      std::shared_lock lk(mx_);
      auto             it = map_.find(Key{id});
      if (it == map_.end()) {
        return std::nullopt;
      }
      return it->second;
    }

    mutable std::shared_mutex               mx_;
    std::unordered_map<Key, Entry, KeyHash> map_;
  };

  // One static Channel<T> per message type T.
  template <class T>
  static Channel<T>& chan()
  {
    static Channel<T> c;
    return c;
  }
};

// ============= Inline impls =============

template <class T>
inline core::Result<void> EventBus::register_queue(EntityId id, size_t capacity, size_t keep_last)
{
  static_assert(!std::is_reference_v<T>);
  return chan<T>().register_queue(id, capacity, keep_last);
}

template <class T>
inline core::Result<void> EventBus::unregister_queue(EntityId id)
{
  return chan<T>().unregister_queue(id);
}

template <class T>
inline bool EventBus::is_registered(EntityId id) const
{
  return chan<T>().is_registered(id);
}

template <class T>
inline core::Result<void> EventBus::publish(EntityId id, T&& msg)
{
  using BaseT = std::remove_cvref_t<T>;
  return chan<BaseT>().publish(id, std::forward<T>(msg));
}

template <typename T>
inline core::Result<void> EventBus::try_dequeue(EntityId id, T& msg)
{
  auto result = chan<T>().try_dequeue(id);
  if (result) {
    msg = std::move(*result);
    return {};
  }
  return std::unexpected(result.error());
}

template <class T>
inline core::Result<T> EventBus::try_dequeue(EntityId id)
{
  return chan<T>().try_dequeue(id);
}

template <class T>
inline void EventBus::clear()
{
  chan<T>().clear();
}

} // namespace ddrl::events
