#ifndef CLONING_HPP
#define CLONING_HPP

#include <cvextra/mats.hpp>
#include "stdx.hpp"

#include <type_traits>
#include <memory>

namespace stdx {

template <typename T>
auto clone(T const& t
		) -> std::unique_ptr<T>
{
    return t.clone();
}

//template <typename T>
//auto move_(T const& t
//        ) -> std::unique_ptr<T>
//{
//    return t.move();
//}

template <typename T>
class cloned_unique_ptr
{
    template <typename U>
    friend class cloned_unique_ptr;

    static_assert(std::is_same<decltype( clone(std::declval<T>()) ), std::unique_ptr<T>>::value,
            "cloned_unique_ptr<T> requires a type for which"
            " the global function 'clone' returns a unique_ptr<T>");

private:
    std::unique_ptr<T> up;

public:
    cloned_unique_ptr() = default;

    cloned_unique_ptr(cloned_unique_ptr const& other)
        : up(other.up ? clone(*other.up) : nullptr) {}

    cloned_unique_ptr(cloned_unique_ptr&& other) : up(std::move(other.up)) {}

    explicit cloned_unique_ptr(T const& other)
        : up(clone(other)) {}

    template <typename U>
    cloned_unique_ptr(cloned_unique_ptr<U> const& other)
        : up(other.up ? clone(static_cast<T const&>(*other.up)) : nullptr) {}

    template <typename U>
    cloned_unique_ptr(cloned_unique_ptr<U>&& other) : up(std::move(other.up)) {}

    template <typename U>
    cloned_unique_ptr(std::unique_ptr<U> _up) : up(std::move(_up)) {}

    template <typename U>
    auto operator =(cloned_unique_ptr<U> const& other) -> cloned_unique_ptr<T>&
    {
        if (other.up)
        {
            up = clone(static_cast<T const&>(*other.up));
        } else
        {
            up.reset();
        }
        return *this;
    }

    template <typename U>
    auto operator =(cloned_unique_ptr<U>&& other) -> cloned_unique_ptr<T>&
    {
        up = std::move(other.up);
        return *this;
    }

    operator bool() const {return (bool)up;}
    auto operator *() const -> decltype(*up) {return *up;}
    auto operator->() const -> decltype(up.operator->()) {return up.operator->();}
    auto get() const -> T* {return up.get();}
};

template <typename T>
class copy_constructed_unique_ptr
{
private:
    std::unique_ptr<T> up;

public:
    copy_constructed_unique_ptr() = default;

    copy_constructed_unique_ptr(copy_constructed_unique_ptr<T> const& other)
        : up(other.up ? stdx::make_unique<T>(*other.up) : nullptr) {}

    copy_constructed_unique_ptr(copy_constructed_unique_ptr<T>&& other) : up(std::move(other.up)) {}

    copy_constructed_unique_ptr(std::unique_ptr<T> _up) : up(std::move(_up)) {}

    auto operator =(copy_constructed_unique_ptr<T> const& other) -> copy_constructed_unique_ptr<T>&
    {
        if (other.up)
        {
            up = stdx::make_unique<T>(*other.up);
        } else
        {
            up.reset();
        }
        return *this;
    }

    auto operator =(copy_constructed_unique_ptr<T>&& other) -> copy_constructed_unique_ptr<T>&
    {
        up = std::move(other.up);
        return *this;
    }

    operator bool() {return up;}
    auto operator *() const -> decltype(*up) {return *up;}
    auto operator->() const -> decltype(up.operator->()) {return up.operator->();}
    auto get() const -> T* {return up.get();}
};


template<typename T, typename ...Args>
auto make_cloned_unique(Args&& ...args) -> stdx::cloned_unique_ptr<T>
{
    return stdx::cloned_unique_ptr<T>(stdx::make_unique<T>(std::forward<Args>(args)...));
}

template<typename T, typename ...Args>
auto make_copy_constructed_unique(Args&& ...args) -> stdx::copy_constructed_unique_ptr<T>
{
    return stdx::copy_constructed_unique_ptr<T>(stdx::make_unique<T>(std::forward<Args>(args)...));
}

} // namespace stdx


// Macros for defining cloning member functions.
// rawClone() is a virtual function that uses covariant return types
// and is *overridden* in the derived classes,
// while clone() is always a newly introduced function in the inheritance
// hierarchy that *hides* the corresponding function in the base class.

#define CVX_CLONE_IN_DERIVED(Derived) \
protected: \
    virtual auto rawClone() const -> Derived* \
    { \
        return new Derived(*this); \
    } \
public: \
    auto clone() const -> std::unique_ptr<Derived> \
    { \
        return std::unique_ptr<Derived>(rawClone()); \
    }

#define CVX_CLONE_IN_BASE(Base) \
protected: \
    virtual auto rawClone() const -> Base* = 0; \
public: \
    auto clone() const -> std::unique_ptr<Base> \
    { \
        return std::unique_ptr<Base>(rawClone()); \
    }

#define CVX_CLONE_IN_SINGLE(Type) \
public: \
    auto clone() const -> std::unique_ptr<Type> \
    { \
        return std::unique_ptr<Type>(new Type(*this)); \
    }



#endif // CLONING_HPP
