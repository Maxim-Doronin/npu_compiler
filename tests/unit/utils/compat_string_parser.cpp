//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/utils/ov/compat_string_parser.hpp"

#include <gtest/gtest.h>

using namespace vpux::compat::parser;

std::unordered_map<std::string, std::string> parseInput(const std::string& input) {
    Parser parser(input);
    return parser.getAttributes();
}

TEST(CompatStringParserTest, SimpleAttributes) {
    const auto attrs = parseInput("key=VALUE;my_id=123");
    EXPECT_EQ(attrs.size(), 2);
    EXPECT_EQ(attrs.at("key"), "VALUE");
    EXPECT_EQ(attrs.at("my_id"), "123");
}

TEST(CompatStringParserTest, DuplicateAttributes) {
    EXPECT_THROW(parseInput("same=1;same=2"), std::runtime_error);
    EXPECT_THROW(parseInput("same=1;{same=2}"), std::runtime_error);
    EXPECT_NO_THROW(parseInput("same=1;list=[same=2]"));
}

TEST(CompatStringParserTest, OptionalAttributes) {
    auto parser = Parser("key=VAL;{optional=NESTED_VAL}");
    const auto& attrs = parser.getAttributes();
    const auto& optionalAttrs = parser.getOptionalAttributes();
    EXPECT_EQ(attrs.size(), 1);
    EXPECT_EQ(optionalAttrs.size(), 1);
    EXPECT_EQ(attrs.at("key"), "VAL");
    EXPECT_EQ(optionalAttrs.at("optional"), "NESTED_VAL");
}

TEST(CompatStringParserTest, NestedOptionalAttributes) {
    auto parser = Parser("{a=1;{b=2};c=3}");
    const auto& attrs = parser.getAttributes();
    const auto& optionalAttrs = parser.getOptionalAttributes();
    EXPECT_EQ(attrs.size(), 0);
    EXPECT_EQ(optionalAttrs.size(), 3);
    EXPECT_EQ(optionalAttrs.at("a"), "1");
    EXPECT_EQ(optionalAttrs.at("b"), "2");
    EXPECT_EQ(optionalAttrs.at("c"), "3");
}

TEST(CompatStringParserTest, ListCaptureOpaque) {
    const auto attrs = parseInput("data=[attr=A|attr=B]");
    ASSERT_TRUE(attrs.count("data"));
    EXPECT_EQ(attrs.at("data"), "[attr=A|attr=B]");
}

TEST(CompatStringParserTest, NestedListsAndBracesInList) {
    const auto attrs = parseInput("config=[type=X|meta=X;{id=1;tags=[a=1|b=2]}]");
    EXPECT_EQ(attrs.at("config"), "[type=X|meta=X;{id=1;tags=[a=1|b=2]}]");
}

TEST(CompatStringParserTest, SplitListHelper) {
    std::string listContent = "[item=1|item=2;extra=[sub=3]|item=[a=1;b=2;{c=[x=1]}]]";
    std::vector<std::string> parts = Parser::splitList(listContent);

    ASSERT_EQ(parts.size(), 3);
    EXPECT_EQ(parts[0], "item=1");
    EXPECT_EQ(parts[1], "item=2;extra=[sub=3]");
    EXPECT_EQ(parts[2], "item=[a=1;b=2;{c=[x=1]}]");
}

TEST(CompatStringParserTest, EmptyBlocks) {
    EXPECT_THROW(parseInput(""), std::runtime_error);
    EXPECT_THROW(parseInput("{}"), std::runtime_error);
    EXPECT_THROW(parseInput("list=[]"), std::runtime_error);
}

TEST(CompatStringParserTest, InvalidString) {
    EXPECT_THROW(parseInput(";"), std::runtime_error);
    EXPECT_THROW(parseInput("key"), std::runtime_error);
    EXPECT_THROW(parseInput("=VALUE"), std::runtime_error);
    EXPECT_THROW(parseInput("inner=[a=1|b=2]|c=3"), std::runtime_error);
    EXPECT_THROW(parseInput("{"), std::runtime_error);

    EXPECT_THROW(parseInput("key=VALUE;"), std::runtime_error);
    EXPECT_THROW(parseInput("key=VALUE;{}"), std::runtime_error);
}
