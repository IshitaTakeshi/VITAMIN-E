// Copyright 2022 Takeshi Ishita

#ifndef INCLUDE_ARRAY_HPP_
#define INCLUDE_ARRAY_HPP_

#include <vector>

template<typename T>
std::vector<std::reference_wrapper<const T>> get_as_references(
  const std::vector<T> & vector,
  const std::vector<int> & indices) {
  std::vector<std::reference_wrapper<const T>> refs;
  for (const int i : indices) {
    refs.push_back(std::cref(vector.at(i)));
  }
  return refs;
}

template<typename T>
std::vector<T> get_by_indices(
  const std::vector<T> & vector,
  const std::vector<int> & indices) {
  std::vector<T> v;
  for (const int i : indices) {
    v.push_back(vector.at(i));
  }
  return v;
}

#endif  // INCLUDE_ARRAY_HPP_
