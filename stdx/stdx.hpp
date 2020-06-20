#ifndef STDX_HPP
#define STDX_HPP

#include <functional>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace stdx
{

auto to_string(double number, int n_decimals_after_point) -> std::string;
auto to_string(bool b) -> std::string;

template<typename T, typename... Args>
auto make_unique(Args&&... args) -> std::unique_ptr<T>
{
    return std::unique_ptr<T>{new T{std::forward<Args>(args)...}};
}

template<typename Derived, typename Base, typename... Args>
auto make_base_unique(Args&&... args) -> std::unique_ptr<Base>
{
    return std::unique_ptr<Base>{new Derived{std::forward<Args>(args)...}};
}

template<typename T>
class any_reference_wrapper {

public:
    any_reference_wrapper(T&& t) : ref(t) {}
    any_reference_wrapper(T& t) : ref(t) {}

    any_reference_wrapper(any_reference_wrapper const& other)
        : ref(other.ref) {}

    operator T&() const {return ref;}
    T& get() const {return ref;}

private:
    T& ref;
};

template<typename T>
auto operator<<(std::ostream& os, stdx::any_reference_wrapper<T> const& ref) -> std::ostream&
{
    os << ref.get();
    return os;
}

template <typename T, typename Ret, typename...  Args>
auto member(T* obj, Ret(T::*pMember)(Args...)) -> std::function<Ret(Args&&...)>
{
    return [=](Args&&... args)
    {
        (obj->*pMember)(std::forward<Args>(args)...);
    };
}


/**
 * Returns an lvalue reference for an rvalue reference
 * Useful if a function wants a reference but we *know* that
 * a particular rvalue reference will also be okay.
 */
template <class T> inline
auto tempref(T&& obj) -> T&
{
    return obj;
}

/**
 * Returns an iterator to the largest element, where
 * largeness is defined by a custom unary functor's result.
 * The functor takes a dereferenced iterator and
 * must return a type that is comparable with operator <.
 */
template<typename ForwardIterator, typename UnaryOperation>
auto max_element_by(
        ForwardIterator first,
        ForwardIterator const last,
        UnaryOperation op) -> ForwardIterator
{
    if (first == last)
    {
        return last;
    }

    ForwardIterator best = first;
    auto bestValue = op(*first);

    while (++first != last)
    {
        auto const& currentValue = op(*first);
        if (bestValue < currentValue)
        {
            best = first;
            bestValue = currentValue;
        }
    }
    return best;
}

/**
 * Returns an iterator to the smallest element, where
 * smallness is defined by a custom unary functor's result.
 * The functor takes a dereferenced iterator and
 * must return a type that is comparable with operator <.
 */
template<typename ForwardIterator, typename UnaryOperation>
auto min_element_by(
        ForwardIterator first,
        ForwardIterator const last,
        UnaryOperation op) -> ForwardIterator
{
    if (first == last)
    {
        return last;
    }

    ForwardIterator best = first;
    auto bestValue = op(*first);

    while (++first != last)
    {
        auto&& currentValue = op(*first);
        if (currentValue < bestValue)
        {
            best = first;
            bestValue = currentValue;
        }
    }
    return best;
}

} // namespace stdx


template<typename T>
auto operator<<(std::ostream& os, std::vector<T> const& vec) -> std::ostream&
{
    os << "{";
    for (int i=0; i<vec.size(); ++i)
    {
        os << vec[i];
        if (i<vec.size()-1)
        {
            os << ", ";
        }
    }
    os << "}";
    return os;
}


#endif // STDX_HPP
