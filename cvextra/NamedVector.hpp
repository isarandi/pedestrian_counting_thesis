#ifndef CVEXTRA_NAMEDVECTOR_HPP_
#define CVEXTRA_NAMEDVECTOR_HPP_


namespace cvx
{

//template <typename T>
//class NamedVector
//{
//public:
//    NamedVector(
//            std::string const& name,
//            std::vector<T> const& elements)
//        : name(name)
//        , elements(elements){}
//
//    NamedVector(std::vector<T> const& elements)
//        : name(makeRandomName()), elements(elements){}
//
//    NamedVector(){}
//W
//    auto rangeFrom(int begin) const -> NamedVector<T>;
//    auto range(int begin, int end) const -> NamedVector<T>;
//    auto range(cv::Range const& range) const -> NamedVector<T>;
//    auto rangeByRatio(double begin, double end) const -> NamedVector<T>;
//
//    auto getDescription() const -> std::string;
//
//    auto size() const -> int {return elements.size();}
//    auto operator [](int i) const -> T const& {return elements[i];}
//    auto begin() const -> std::vector<T>::const_iterator {return elements.begin();}
//    auto end() const -> std::vector<T>::const_iterator {return elements.end();}
//
//    auto getFrames() const -> std::vector<T> const& {return elements;}
//    auto getName() const -> std::string {return name;}
//
//    void append(NamedVector<T> const& other);
//
//private:
//    static auto makeRandomName() -> std::string;
//
//    std::vector<T> elements;
//    std::string name;
//};

} /* namespace cvx */


#endif /* CVEXTRA_NAMEDVECTOR_HPP_ */
