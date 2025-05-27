#pragma once

#include <coroutine>
#include <optional>

// A class created for the single purpose of replacing the <generator>
// header which is only available in C++/23.
template <typename T>
class generator
{
public:
    struct promise_type
    {
        std::optional<T> current_value;

        generator get_return_object()
        {
            return generator{
                std::coroutine_handle<promise_type>::from_promise(*this)};
        }
        std::suspend_always initial_suspend() { return {}; }
        std::suspend_always final_suspend() noexcept { return {}; }
        std::suspend_always yield_value(T value)
        {
            current_value = std::move(value);
            return {};
        }
        void return_void() {}
        void unhandled_exception()
        {
            throw;
        }
    };

    struct iterator
    {
        std::coroutine_handle<promise_type> coro;

        iterator(std::coroutine_handle<promise_type> coro) : coro(coro) {}

        iterator &operator++()
        {
            coro.resume();
            if (coro.done())
                coro = nullptr;
            return *this;
        }

        const T &operator*() const
        {
            return *coro.promise().current_value;
        }

        bool operator==(std::default_sentinel_t) const
        {
            return !coro || coro.done();
        }

        bool operator!=(std::default_sentinel_t s) const
        {
            return !(*this == s);
        }
    };

    using sentinel = std::default_sentinel_t;

    generator(std::coroutine_handle<promise_type> h) : coro(h) {}
    generator(generator &&other) noexcept : coro(other.coro) { other.coro = {}; }
    ~generator()
    {
        if (coro)
            coro.destroy();
    }

    iterator begin()
    {
        if (coro)
            coro.resume();
        return iterator{coro};
    }
    sentinel end() { return {}; }

private:
    std::coroutine_handle<promise_type> coro;
};
