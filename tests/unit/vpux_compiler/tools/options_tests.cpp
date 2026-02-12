//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/tools/options.hpp"
#include "vpux/utils/core/string_ref.hpp"

#include <gtest/gtest.h>

using namespace vpux;

TEST(ParseArchKindTest, NoArchKind) {
    const char* noOptions[] = {
            "./vpux-opt",
    };
    try {
        std::ignore = vpux::parseArchKind(std::size(noOptions), const_cast<char**>(noOptions));
        GTEST_FAIL() << "Exception must be thrown";
    } catch (const std::exception& e) {
        EXPECT_TRUE(StringRef(e.what()).contains("Can't get ArchKind value"));
    }

    const char* noArchKind[] = {
            "./vpux-opt",
            "--init-compiler=allow-something-else",
            "foo.mlir",
    };
    try {
        std::ignore = vpux::parseArchKind(std::size(noArchKind), const_cast<char**>(noArchKind));
        GTEST_FAIL() << "Exception must be thrown";
    } catch (const std::exception& e) {
        EXPECT_TRUE(StringRef(e.what()).contains("Can't get ArchKind value"));
    }
}

TEST(ParseArchKindTest, NoArchKindWithHelp) {
    const char* noOptions[] = {
            "./vpux-opt",
            "-h",
    };
    EXPECT_FALSE(vpux::parseArchKind(std::size(noOptions), const_cast<char**>(noOptions)).has_value());

    const char* noArchKind[] = {
            "./vpux-opt",
            "--init-compiler=allow-something-else",
            "foo.mlir",
            "--help",
    };
    EXPECT_FALSE(vpux::parseArchKind(std::size(noArchKind), const_cast<char**>(noArchKind)).has_value());
}

TEST(ParseArchKindTest, SpecifiedMoreThanOnce) {
    const char* redefinition[] = {
            "./vpux-opt",
            "--vpu-arch=NPU40XX",
            "foo.mlir",
            "--init-compiler=vpu-arch=NPU40XX",
    };
    try {
        std::ignore = vpux::parseArchKind(std::size(redefinition), const_cast<char**>(redefinition));
        GTEST_FAIL() << "Exception must be thrown";
    } catch (const std::exception& e) {
        EXPECT_TRUE(StringRef(e.what()).contains("ArchKind value is ambiguous"));
    }

    // Note: this still fails
    const char* redefinitionWithHelp[] = {
            "./vpux-opt", "--vpu-arch=NPU40XX", "--help", "foo.mlir", "--init-compiler=vpu-arch=NPU40XX",
    };
    try {
        std::ignore = vpux::parseArchKind(std::size(redefinitionWithHelp), const_cast<char**>(redefinitionWithHelp));
        GTEST_FAIL() << "Exception must be thrown";
    } catch (const std::exception& e) {
        EXPECT_TRUE(StringRef(e.what()).contains("ArchKind value is ambiguous"));
    }
}

TEST(ParseArchKindTest, InvalidArchKind) {
    const char* unknownArchKind[] = {
            "./vpux-opt",
            "vpu-arch=NPU18XX",
    };
    try {
        std::ignore = vpux::parseArchKind(std::size(unknownArchKind), const_cast<char**>(unknownArchKind));
        GTEST_FAIL() << "Exception must be thrown";
    } catch (const std::exception& e) {
        EXPECT_TRUE(StringRef(e.what()).contains("Unknown VPU architecture"));
    }

    const char* badValueArchKind[] = {
            "./vpux-opt",
            "vpu-arch=NPU40XXnoisytail",
    };
    try {
        std::ignore = vpux::parseArchKind(std::size(badValueArchKind), const_cast<char**>(badValueArchKind));
        GTEST_FAIL() << "Exception must be thrown";
    } catch (const std::exception& e) {
        EXPECT_TRUE(StringRef(e.what()).contains("Unknown VPU architecture"));
    }

    const char* unknownArchKindWithHelp[] = {
            "./vpux-opt",
            "vpu-arch=NPU18XX",
            "-h",
    };
    EXPECT_FALSE(vpux::parseArchKind(std::size(unknownArchKindWithHelp), const_cast<char**>(unknownArchKindWithHelp))
                         .has_value());
}

TEST(ParseArchKindTest, VpuArchSpecified) {
    const char* archInMiddle[] = {
            "./vpux-opt",
            "--split-input-file",
            "--vpu-arch=NPU40XX",
            "foo.mlir",
    };
    EXPECT_EQ(config::ArchKind::NPU40XX,
              vpux::parseArchKind(std::size(archInMiddle), const_cast<char**>(archInMiddle)).value());

    const char* archLast[] = {
            "./vpux-opt",
            "--split-input-file",
            "foo.mlir",
            "--vpu-arch=NPU37XX",
    };
    EXPECT_EQ(config::ArchKind::NPU37XX,
              vpux::parseArchKind(std::size(archLast), const_cast<char**>(archLast)).value());

    const char* archWithNoise[] = {
            "./vpux-opt", "--split-input-file",
            "--vpu-arch=NPU37XX foo.mlir",  // e.g. two arguments are squashed together
    };
    EXPECT_EQ(config::ArchKind::NPU37XX,
              vpux::parseArchKind(std::size(archWithNoise), const_cast<char**>(archWithNoise)).value());

    const char* archWithHelp[] = {
            "./vpux-opt", "--help", "--split-input-file", "foo.mlir", "--vpu-arch=NPU37XX",
    };
    EXPECT_EQ(config::ArchKind::NPU37XX,
              vpux::parseArchKind(std::size(archWithHelp), const_cast<char**>(archWithHelp)).value());
}

TEST(ParseArchKindTest, InitCompilerSpecified) {
    const char* justArch[] = {
            "./vpux-opt",
            "--split-input-file",
            "--init-compiler=vpu-arch=NPU40XX",
            "foo.mlir",
    };
    EXPECT_EQ(config::ArchKind::NPU40XX,
              vpux::parseArchKind(std::size(justArch), const_cast<char**>(justArch)).value());

    const char* moreThanArch[] = {
            "./vpux-opt",
            "--split-input-file",
            "--init-compiler=some-other-option=X vpu-arch=NPU40XX allow-custom-values=true",
            "foo.mlir",
    };
    EXPECT_EQ(config::ArchKind::NPU40XX,
              vpux::parseArchKind(std::size(moreThanArch), const_cast<char**>(moreThanArch)).value());

    const char* archWithHelp[] = {
            "./vpux-opt", "--split-input-file", "--init-compiler=vpu-arch=NPU40XX", "foo.mlir", "--help",
    };
    EXPECT_EQ(config::ArchKind::NPU40XX,
              vpux::parseArchKind(std::size(archWithHelp), const_cast<char**>(archWithHelp)).value());
}
