// Copyright 2022 Takeshi Ishita

#include <gtest/gtest.h>

#include "vector.hpp"


struct S {
  int i;
};

TEST(GetAsReferences, SmokeTest)
{
  const std::vector<S> structs = {S{1}, S{2}, S{3}};

  EXPECT_EQ(get_as_references(structs, std::vector<int>{}).size(), 0);

  const auto refs = get_as_references(structs, std::vector<int>{1, 2});

  EXPECT_EQ(refs.size(), 2);

  EXPECT_EQ(refs.at(0).get().i, 2);
  EXPECT_EQ(refs.at(1).get().i, 3);
}

TEST(GetByIndices, SmokeTest)
{
  const std::vector<S> structs = {S{1}, S{2}, S{3}};

  EXPECT_EQ(get_by_indices(structs, std::vector<int>{}).size(), 0);

  const auto v = get_by_indices(structs, std::vector<int>{1, 2});

  EXPECT_EQ(v.size(), 2);

  EXPECT_EQ(v.at(0).i, 2);
  EXPECT_EQ(v.at(1).i, 3);
}
