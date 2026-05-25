//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/*
    This file implements a simple parser for compatibility strings used in the
    NPU software stack. The compatibility string encodes requirements of a
    compiled blob and version metadata, such as target platform, requirements
    and compiler version. The string consists of human readable list of
    name=value pairs called attributes. Basic encapsulation across SW layers is
    supported via nested lists. Optional attributes are supported with brace
    enclosed attribute list.

    The grammar for the compatibility string is defined as follows:

        string  ::=  expr (';' expr)*
        expr    ::=  attr | '{' string '}'
        attr    ::=  name '=' value | name '=' list
        list    ::=  '[' string ('|' string)* ']'
        name    ::=  [a-z][_a-z0-9]*
        value   ::=  [A-Z0-9][_A-Z0-9\.]*

    No whitespaces are allowed.
    Attribute names are lower case. Attribute values are upper case.
    Attribute names SHOULD be globally unique.

    Example usage:

        Parser parser("name=A;nested=[key=X;ver=1.2|key=Y;ver=1.3];{optional=V1.4}");
        parser.getAttribute("name");                  // returns "A"
        auto nested = parser.getAttribute("nested");  // returns "[key=X;ver=1.2|key=Y;ver=1.3]"
        Parser::splitList(nested);                    // returns ["key=X;ver=1.2", "key=Y;ver=1.3"]
                                                      // to be parsed separately
        parser.getOptionalAttributes();               // returns { "optional": "V1.4"} as map
*/

#pragma once

#include <cassert>
#include <cctype>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace vpux::compat::parser {

namespace detail {

enum class TokenType { NAME, VALUE, EQUALS, SEMICOLON, LBRACE, RBRACE, LBRACKET, RBRACKET, PIPE, END };

struct Token {
    TokenType type;
    std::string value;
};

class Lexer {
    std::string_view input;
    size_t pos = 0;

public:
    Lexer(std::string_view src): input(src) {
    }

    Token nextToken() {
        if (pos >= input.length()) {
            return {TokenType::END, ""};
        }

        unsigned char current = input[pos];

        // clang-format off
        if (current == '=') { pos++; return {TokenType::EQUALS, "="}; }
        if (current == ';') { pos++; return {TokenType::SEMICOLON, ";"}; }
        if (current == '{') { pos++; return {TokenType::LBRACE, "{"}; }
        if (current == '}') { pos++; return {TokenType::RBRACE, "}"}; }
        if (current == '[') { pos++; return {TokenType::LBRACKET, "["}; }
        if (current == ']') { pos++; return {TokenType::RBRACKET, "]"}; }
        if (current == '|') { pos++; return {TokenType::PIPE, "|"}; }
        // clang-format on

        // Match Name: [a-z][_a-z0-9]*
        if (islower(current)) {
            std::string s;
            while (pos < input.length() && (islower(static_cast<unsigned char>(input[pos])) ||
                                            isdigit(static_cast<unsigned char>(input[pos])) || input[pos] == '_')) {
                s += input[pos++];
            }
            return {TokenType::NAME, std::move(s)};
        }

        // Match Value: [A-Z0-9][_A-Z0-9\.]*
        if (isupper(current) || isdigit(current)) {
            std::string s;
            while (pos < input.length() &&
                   (isupper(static_cast<unsigned char>(input[pos])) ||
                    isdigit(static_cast<unsigned char>(input[pos])) || input[pos] == '.' || input[pos] == '_')) {
                s += input[pos++];
            }
            return {TokenType::VALUE, std::move(s)};
        }

        throw std::runtime_error("Unexpected character: " + std::string(1, current));
    }
};

}  // namespace detail

class Parser {
    using Lexer = detail::Lexer;
    using Token = detail::Token;
    using TokenType = detail::TokenType;

    Lexer _lexer;
    Token _next;
    std::unordered_map<std::string, std::string> _attributes;
    std::unordered_map<std::string, std::string> _optional_attributes;
    int _nesting = 0;
    std::string _captureBuffer;  // Used to return the content of a list as an attribute
    int _optional = 0;           // Tracks top-level optional attributes

public:
    Parser(std::string_view str): _lexer(str) {
        _next = _lexer.nextToken();
        parseString();
        if (_next.type != TokenType::END) {
            throw std::runtime_error("Unexpected trailing characters in compatibility string");
        }
    }

    const std::string& getAttribute(const std::string& name) const {
        auto it = _attributes.find(name);
        if (it != _attributes.end()) {
            return it->second;
        }
        throw std::runtime_error("Attribute not found: " + name);
    }

    const std::unordered_map<std::string, std::string>& getAttributes() const {
        return _attributes;
    }
    const std::unordered_map<std::string, std::string>& getOptionalAttributes() const {
        return _optional_attributes;
    }

    // Breaks down a list of nested strings. To be used on parser output only
    static std::vector<std::string> splitList(const std::string& content) {
        std::vector<std::string> parts;
        std::string current;

        assert(!content.empty() && content.front() == '[' && content.back() == ']');
        std::string_view inner(content.c_str() + 1, content.size() - 2);  // Strip []

        int depth = 0;
        for (char c : inner) {
            if (c == '[' || c == '{') {
                depth++;
            }
            if (c == ']' || c == '}') {
                depth--;
            }
            if (c == '|' && depth == 0) {
                parts.push_back(current);
                current.clear();
            } else {
                current += c;
            }
        }
        assert(depth == 0);
        parts.push_back(std::move(current));
        return parts;
    }

private:
    std::string consume(TokenType type) {
        if (_next.type != type) {
            throw std::runtime_error("Unexpected token: " + _next.value);
        }
        Token current = _next;
        _next = _lexer.nextToken();
        if (_nesting > 0) {
            _captureBuffer += current.value;
            return "<NULL>";  // ignored
        } else {
            return current.value;
        }
    }

    // string ::= expr (';' expr)*
    void parseString() {
        parseExpr();
        while (_next.type == TokenType::SEMICOLON) {
            consume(TokenType::SEMICOLON);
            parseExpr();
        }
    }

    // expr ::= attr | '{' string '}'
    void parseExpr() {
        if (_next.type == TokenType::LBRACE) {
            if (_nesting == 0) {
                ++_optional;
            }
            consume(TokenType::LBRACE);
            parseString();
            consume(TokenType::RBRACE);
            if (_nesting == 0) {
                --_optional;
            }
        } else {
            parseAttr();
        }
    }

    // attr ::= name '=' value | name '=' list
    void parseAttr() {
        auto name = consume(TokenType::NAME);
        consume(TokenType::EQUALS);
        if (_next.type == TokenType::LBRACKET) {
            ++_nesting;
            parseList();
            --_nesting;
            if (_nesting == 0) {
                setAttribute(name, _captureBuffer);
                _captureBuffer.clear();
            }
            return;
        }
        auto value = consume(TokenType::VALUE);
        if (_nesting == 0) {
            setAttribute(name, value);
        }
    }

    // list ::= '[' string ('|' string)* ']'
    void parseList() {
        consume(TokenType::LBRACKET);
        parseString();
        while (_next.type == TokenType::PIPE) {
            consume(TokenType::PIPE);
            parseString();
        }
        consume(TokenType::RBRACKET);
    }

    void setAttribute(const std::string& name, const std::string& value) {
        if (_attributes.count(name) != 0 || _optional_attributes.count(name) != 0) {
            throw std::runtime_error("Duplicate attribute: " + name);
        }
        if (_optional > 0) {
            _optional_attributes[name] = value;
        } else {
            _attributes[name] = value;
        }
    }
};

}  // namespace vpux::compat::parser
