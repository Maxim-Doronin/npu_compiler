//
// Copyright (C) 2025-2026 Intel Corporation
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
        std::ignore = vpux::parseParamsAndDeduceArch(std::size(noOptions), const_cast<char**>(noOptions));
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
        std::ignore = vpux::parseParamsAndDeduceArch(std::size(noArchKind), const_cast<char**>(noArchKind));
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
    EXPECT_FALSE(vpux::parseParamsAndDeduceArch(std::size(noOptions), const_cast<char**>(noOptions)).has_value());

    const char* noArchKind[] = {
            "./vpux-opt",
            "--init-compiler=allow-something-else",
            "foo.mlir",
            "--help",
    };
    EXPECT_FALSE(vpux::parseParamsAndDeduceArch(std::size(noArchKind), const_cast<char**>(noArchKind)).has_value());
}

TEST(ParseArchKindTest, SpecifiedMoreThanOnce) {
    const char* redefinition[] = {
            "./vpux-opt",
            "--vpu-arch=NPU40XX",
            "foo.mlir",
            "--init-compiler=vpu-arch=NPU40XX",
    };
    try {
        std::ignore = vpux::parseParamsAndDeduceArch(std::size(redefinition), const_cast<char**>(redefinition));
        GTEST_FAIL() << "Exception must be thrown";
    } catch (const std::exception& e) {
        EXPECT_TRUE(StringRef(e.what()).contains("ArchKind value is ambiguous"));
    }

    // Note: this still fails
    const char* redefinitionWithHelp[] = {
            "./vpux-opt", "--vpu-arch=NPU40XX", "--help", "foo.mlir", "--init-compiler=vpu-arch=NPU40XX",
    };
    try {
        std::ignore = vpux::parseParamsAndDeduceArch(std::size(redefinitionWithHelp),
                                                     const_cast<char**>(redefinitionWithHelp));
        GTEST_FAIL() << "Exception must be thrown";
    } catch (const std::exception& e) {
        EXPECT_TRUE(StringRef(e.what()).contains("ArchKind value is ambiguous"));
    }
}

TEST(ParseArchKindTest, InvalidArchKind) {
    const char* unknownArchKind[] = {
            "./vpux-opt",
            "--vpu-arch=NPU18XX",
    };
    try {
        std::ignore = vpux::parseParamsAndDeduceArch(std::size(unknownArchKind), const_cast<char**>(unknownArchKind));
        GTEST_FAIL() << "Exception must be thrown";
    } catch (const std::exception& e) {
        EXPECT_TRUE(StringRef(e.what()).contains("Unknown VPU architecture"));
    }

    const char* badValueArchKind[] = {
            "./vpux-opt",
            "--vpu-arch=NPU40XXnoisytail",
    };
    try {
        std::ignore = vpux::parseParamsAndDeduceArch(std::size(badValueArchKind), const_cast<char**>(badValueArchKind));
        GTEST_FAIL() << "Exception must be thrown";
    } catch (const std::exception& e) {
        EXPECT_TRUE(StringRef(e.what()).contains("Unknown VPU architecture"));
    }

    const char* unknownArchKindWithHelp[] = {
            "./vpux-opt",
            "--vpu-arch=NPU18XX",
            "-h",
    };
    EXPECT_FALSE(vpux::parseParamsAndDeduceArch(std::size(unknownArchKindWithHelp),
                                                const_cast<char**>(unknownArchKindWithHelp))
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
              vpux::parseParamsAndDeduceArch(std::size(archInMiddle), const_cast<char**>(archInMiddle)).value());

    const char* archLast[] = {
            "./vpux-opt",
            "--split-input-file",
            "foo.mlir",
            "--vpu-arch=NPU37XX",
    };
    EXPECT_EQ(config::ArchKind::NPU37XX,
              vpux::parseParamsAndDeduceArch(std::size(archLast), const_cast<char**>(archLast)).value());

    const char* archWithNoise[] = {
            "./vpux-opt", "--split-input-file",
            "--vpu-arch=NPU37XX foo.mlir",  // e.g. two arguments are squashed together
    };
    EXPECT_EQ(config::ArchKind::NPU37XX,
              vpux::parseParamsAndDeduceArch(std::size(archWithNoise), const_cast<char**>(archWithNoise)).value());

    const char* archWithHelp[] = {
            "./vpux-opt", "--help", "--split-input-file", "foo.mlir", "--vpu-arch=NPU37XX",
    };
    EXPECT_EQ(config::ArchKind::NPU37XX,
              vpux::parseParamsAndDeduceArch(std::size(archWithHelp), const_cast<char**>(archWithHelp)).value());
}

TEST(ParseArchKindTest, InitCompilerSpecified) {
    const char* justPlatform[] = {
            "./vpux-opt",
            "--split-input-file",
            "--init-compiler=platform=NPU4000",
            "foo.mlir",
    };
    EXPECT_EQ(config::ArchKind::NPU40XX,
              vpux::parseParamsAndDeduceArch(std::size(justPlatform), const_cast<char**>(justPlatform)).value());

    const char* moreThanPlatform[] = {
            "./vpux-opt",         "--split-input-file",       "--init-compiler=some-other-option=X",
            "--platform=NPU4000", "allow-custom-values=true", "foo.mlir",
    };
    EXPECT_EQ(
            config::ArchKind::NPU40XX,
            vpux::parseParamsAndDeduceArch(std::size(moreThanPlatform), const_cast<char**>(moreThanPlatform)).value());

    const char* platformWithHelp[] = {
            "./vpux-opt", "--split-input-file", "--init-compiler=platform=NPU4000", "foo.mlir", "--help",
    };
    EXPECT_EQ(
            config::ArchKind::NPU40XX,
            vpux::parseParamsAndDeduceArch(std::size(platformWithHelp), const_cast<char**>(platformWithHelp)).value());
}

TEST(ParsePlatformTest, SpecifiedMoreThanOnce) {
    const char* redefinition[] = {
            "./vpux-opt",
            "--platform=NPU4000",
            "foo.mlir",
            "--init-compiler=platform=NPU4000",
    };
    try {
        std::ignore = vpux::parseParamsAndDeduceArch(std::size(redefinition), const_cast<char**>(redefinition));
        GTEST_FAIL() << "Exception must be thrown";
    } catch (const std::exception& e) {
        EXPECT_TRUE(StringRef(e.what()).contains("Platform value is ambiguous"));
    }

    // Note: this still fails
    const char* redefinitionWithHelp[] = {
            "./vpux-opt", "--platform=NPU4000", "--help", "foo.mlir", "--init-compiler=platform=NPU4000",
    };
    try {
        std::ignore = vpux::parseParamsAndDeduceArch(std::size(redefinitionWithHelp),
                                                     const_cast<char**>(redefinitionWithHelp));
        GTEST_FAIL() << "Exception must be thrown";
    } catch (const std::exception& e) {
        EXPECT_TRUE(StringRef(e.what()).contains("Platform value is ambiguous"));
    }
}

TEST(ParsePlatformTest, PlatformAndArchSpecified) {
    const char* platformAndArch[] = {
            "./vpux-opt",
            "--platform=NPU4000",
            "--vpu-arch=NPU40XX",
            "foo.mlir",
    };
    try {
        std::ignore = vpux::parseParamsAndDeduceArch(std::size(platformAndArch), const_cast<char**>(platformAndArch));
        GTEST_FAIL() << "Exception must be thrown";
    } catch (const std::exception& e) {
        EXPECT_TRUE(StringRef(e.what()).contains("Can't use \"platform\" and \"vpu-arch\" at the same time"));
    }

    const char* platformAndInitArch[] = {
            "./vpux-opt",
            "--platform=NPU4000",
            "--init-compiler=vpu-arch=NPU40XX",
            "foo.mlir",
    };
    try {
        std::ignore =
                vpux::parseParamsAndDeduceArch(std::size(platformAndInitArch), const_cast<char**>(platformAndInitArch));
        GTEST_FAIL() << "Exception must be thrown";
    } catch (const std::exception& e) {
        EXPECT_TRUE(StringRef(e.what()).contains("Can't use \"platform\" and \"vpu-arch\" at the same time"));
    }

    const char* initPlatformAndArch[] = {
            "./vpux-opt",
            "--init-compiler=platform=NPU4000",
            "--vpu-arch=NPU40XX",
            "foo.mlir",
    };
    try {
        std::ignore =
                vpux::parseParamsAndDeduceArch(std::size(initPlatformAndArch), const_cast<char**>(initPlatformAndArch));
        GTEST_FAIL() << "Exception must be thrown";
    } catch (const std::exception& e) {
        EXPECT_TRUE(StringRef(e.what()).contains("Can't use \"platform\" and \"vpu-arch\" at the same time"));
    }
}

TEST(ParsePlatformTest, InvalidPlatform) {
    const char* unknownPlatform[] = {
            "./vpux-opt",
            "--platform=NPU18XX",
    };
    try {
        std::ignore = vpux::parseParamsAndDeduceArch(std::size(unknownPlatform), const_cast<char**>(unknownPlatform));
        GTEST_FAIL() << "Exception must be thrown";
    } catch (const std::exception& e) {
        EXPECT_TRUE(StringRef(e.what()).contains("Can't get ArchKind value"));
    }

    const char* badValuePlatform[] = {
            "./vpux-opt",
            "--platform=NPU40XXaaa",
    };
    try {
        std::ignore = vpux::parseParamsAndDeduceArch(std::size(badValuePlatform), const_cast<char**>(badValuePlatform));
        GTEST_FAIL() << "Exception must be thrown";
    } catch (const std::exception& e) {
        EXPECT_TRUE(StringRef(e.what()).contains("Can't get ArchKind value"));
    }

    const char* unknownPlatformWithHelp[] = {
            "./vpux-opt",
            "--platform=NPU18XX",
            "-h",
    };
    EXPECT_FALSE(vpux::parseParamsAndDeduceArch(std::size(unknownPlatformWithHelp),
                                                const_cast<char**>(unknownPlatformWithHelp))
                         .has_value());
}

TEST(ParsePlatformTest, PlatformSpecified) {
    const char* platformInMiddle[] = {
            "./vpux-opt",
            "--split-input-file",
            "--platform=NPU4000",
            "foo.mlir",
    };
    EXPECT_EQ(
            config::ArchKind::NPU40XX,
            vpux::parseParamsAndDeduceArch(std::size(platformInMiddle), const_cast<char**>(platformInMiddle)).value());

    const char* platformLast[] = {
            "./vpux-opt",
            "--split-input-file",
            "foo.mlir",
            "--platform=NPU3720",
    };
    EXPECT_EQ(config::ArchKind::NPU37XX,
              vpux::parseParamsAndDeduceArch(std::size(platformLast), const_cast<char**>(platformLast)).value());

    const char* platformWithNoise[] = {
            "./vpux-opt", "--split-input-file",
            "--platform=NPU3720 foo.mlir",  // e.g. two arguments are squashed together
    };
    EXPECT_EQ(config::ArchKind::NPU37XX,
              vpux::parseParamsAndDeduceArch(std::size(platformWithNoise), const_cast<char**>(platformWithNoise))
                      .value());

    const char* platformWithHelp[] = {
            "./vpux-opt", "--help", "--split-input-file", "foo.mlir", "--platform=NPU3720",
    };
    EXPECT_EQ(
            config::ArchKind::NPU37XX,
            vpux::parseParamsAndDeduceArch(std::size(platformWithHelp), const_cast<char**>(platformWithHelp)).value());
}

TEST(ParsePlatformTest, InitCompilerSpecified) {
    const char* justPlatform[] = {
            "./vpux-opt",
            "--split-input-file",
            "--init-compiler=platform=NPU4000",
            "foo.mlir",
    };
    EXPECT_EQ(config::ArchKind::NPU40XX,
              vpux::parseParamsAndDeduceArch(std::size(justPlatform), const_cast<char**>(justPlatform)).value());

    const char* moreThanPlatform[] = {
            "./vpux-opt",         "--split-input-file",       "--init-compiler=some-other-option=X",
            "--platform=NPU4000", "allow-custom-values=true", "foo.mlir",
    };
    EXPECT_EQ(
            config::ArchKind::NPU40XX,
            vpux::parseParamsAndDeduceArch(std::size(moreThanPlatform), const_cast<char**>(moreThanPlatform)).value());

    const char* platformWithHelp[] = {
            "./vpux-opt", "--split-input-file", "--init-compiler=platform=NPU4000", "foo.mlir", "--help",
    };
    EXPECT_EQ(
            config::ArchKind::NPU40XX,
            vpux::parseParamsAndDeduceArch(std::size(platformWithHelp), const_cast<char**>(platformWithHelp)).value());
}

TEST(ParsePlatformTest, PlatformWithIncorrectSuffixOrPrefix) {
    const char* initCompilerPlatformWithPrefix[] = {
            "./vpux-opt",
            "--split-input-file",
            "--init-compiler=AAAplatform=NPU4000",
            "foo.mlir",
    };
    try {
        std::ignore = vpux::parseParamsAndDeduceArch(std::size(initCompilerPlatformWithPrefix),
                                                     const_cast<char**>(initCompilerPlatformWithPrefix));
        GTEST_FAIL() << "Exception must be thrown";
    } catch (const std::exception& e) {
        EXPECT_TRUE(StringRef(e.what()).contains("Can't get ArchKind value"));
    }

    const char* initCompilerPlatformWithSuffix[] = {
            "./vpux-opt",
            "--split-input-file",
            "--init-compiler=platformAAA=NPU4000",
            "foo.mlir",
    };
    try {
        std::ignore = vpux::parseParamsAndDeduceArch(std::size(initCompilerPlatformWithSuffix),
                                                     const_cast<char**>(initCompilerPlatformWithSuffix));
        GTEST_FAIL() << "Exception must be thrown";
    } catch (const std::exception& e) {
        EXPECT_TRUE(StringRef(e.what()).contains("Can't get ArchKind value"));
    }

    const char* platformWithSuffix[] = {
            "./vpux-opt",
            "--split-input-file",
            "--platformAAA=NPU4000",
            "foo.mlir",
    };
    try {
        std::ignore =
                vpux::parseParamsAndDeduceArch(std::size(platformWithSuffix), const_cast<char**>(platformWithSuffix));
        GTEST_FAIL() << "Exception must be thrown";
    } catch (const std::exception& e) {
        EXPECT_TRUE(StringRef(e.what()).contains("Can't get ArchKind value"));
    }

    const char* platformWithPrefix[] = {
            "./vpux-opt",
            "--split-input-file",
            "--AAAplatform=NPU4000",
            "foo.mlir",
    };
    try {
        std::ignore =
                vpux::parseParamsAndDeduceArch(std::size(platformWithPrefix), const_cast<char**>(platformWithPrefix));
        GTEST_FAIL() << "Exception must be thrown";
    } catch (const std::exception& e) {
        EXPECT_TRUE(StringRef(e.what()).contains("Can't get ArchKind value"));
    }
}

TEST(ParsePlatformTest, PlatformWithEmptyValue) {
    const char* emptyInitCompiler[] = {
            "./vpux-opt",
            "--split-input-file",
            "--init-compiler=",
            "foo.mlir",
    };
    try {
        std::ignore =
                vpux::parseParamsAndDeduceArch(std::size(emptyInitCompiler), const_cast<char**>(emptyInitCompiler));
        GTEST_FAIL() << "Exception must be thrown";
    } catch (const std::exception& e) {
        EXPECT_TRUE(StringRef(e.what()).contains("Can't get ArchKind value"));
    }

    const char* emptyPlatform[] = {
            "./vpux-opt",
            "--split-input-file",
            "--platform=",
            "foo.mlir",
    };
    try {
        std::ignore = vpux::parseParamsAndDeduceArch(std::size(emptyPlatform), const_cast<char**>(emptyPlatform));
        GTEST_FAIL() << "Exception must be thrown";
    } catch (const std::exception& e) {
        EXPECT_TRUE(StringRef(e.what()).contains("Can't get ArchKind value"));
    }
}

TEST(ParsePlatformTest, InitCompilerWithMultipleOptions) {
    const char* platformWithOtherOptions[] = {
            "./vpux-opt",
            "--split-input-file",
            "--init-compiler=someflag=aaa platform=NPU4000",
            "foo.mlir",
    };
    EXPECT_EQ(config::ArchKind::NPU40XX, vpux::parseParamsAndDeduceArch(std::size(platformWithOtherOptions),
                                                                        const_cast<char**>(platformWithOtherOptions))
                                                 .value());

    const char* platformBeforeOtherOptions[] = {
            "./vpux-opt",
            "--split-input-file",
            "--init-compiler=platform=NPU4000 someflag=aaa",
            "foo.mlir",
    };
    EXPECT_EQ(config::ArchKind::NPU40XX, vpux::parseParamsAndDeduceArch(std::size(platformBeforeOtherOptions),
                                                                        const_cast<char**>(platformBeforeOtherOptions))
                                                 .value());

    const char* platformInMiddle[] = {
            "./vpux-opt",
            "--split-input-file",
            "--init-compiler=flag1=value1 platform=NPU3720 flag2=value2",
            "foo.mlir",
    };
    EXPECT_EQ(
            config::ArchKind::NPU37XX,
            vpux::parseParamsAndDeduceArch(std::size(platformInMiddle), const_cast<char**>(platformInMiddle)).value());
}

TEST(ParsePlatformTest, InitCompilerWithQuotedMultipleOptions) {
    const char* platformWithQuotes[] = {
            "./vpux-opt",
            "--split-input-file",
            "--init-compiler=\"someflag=aaa platform=NPU4000\"",
            "foo.mlir",
    };
    EXPECT_EQ(config::ArchKind::NPU40XX,
              vpux::parseParamsAndDeduceArch(std::size(platformWithQuotes), const_cast<char**>(platformWithQuotes))
                      .value());

    const char* platformBeforeOtherOptionsQuoted[] = {
            "./vpux-opt",
            "--split-input-file",
            "--init-compiler=\"platform=NPU4000 compilation-mode=DefaultHW\"",
            "foo.mlir",
    };
    EXPECT_EQ(config::ArchKind::NPU40XX,
              vpux::parseParamsAndDeduceArch(std::size(platformBeforeOtherOptionsQuoted),
                                             const_cast<char**>(platformBeforeOtherOptionsQuoted))
                      .value());

    const char* platformInMiddleQuoted[] = {
            "./vpux-opt",
            "--split-input-file",
            "--init-compiler=\"flag1=value1 platform=NPU3720 flag2=value2\"",
            "foo.mlir",
    };
    EXPECT_EQ(config::ArchKind::NPU37XX, vpux::parseParamsAndDeduceArch(std::size(platformInMiddleQuoted),
                                                                        const_cast<char**>(platformInMiddleQuoted))
                                                 .value());
}

TEST(ParsePlatformTest, InitResourcesWithMultipleOptions) {
    const char* platformWithOtherOptions[] = {
            "./vpux-opt",
            "--split-input-file",
            "--init-resources=someflag=aaa platform=NPU4000",
            "foo.mlir",
    };
    EXPECT_EQ(config::ArchKind::NPU40XX, vpux::parseParamsAndDeduceArch(std::size(platformWithOtherOptions),
                                                                        const_cast<char**>(platformWithOtherOptions))
                                                 .value());

    const char* platformBeforeOtherOptions[] = {
            "./vpux-opt",
            "--split-input-file",
            "--init-resources=platform=NPU3720 resource-option=value",
            "foo.mlir",
    };
    EXPECT_EQ(config::ArchKind::NPU37XX, vpux::parseParamsAndDeduceArch(std::size(platformBeforeOtherOptions),
                                                                        const_cast<char**>(platformBeforeOtherOptions))
                                                 .value());

    const char* platformInMiddle[] = {
            "./vpux-opt",
            "--split-input-file",
            "--init-resources=flag1=value1 platform=NPU4000 flag2=value2",
            "foo.mlir",
    };
    EXPECT_EQ(
            config::ArchKind::NPU40XX,
            vpux::parseParamsAndDeduceArch(std::size(platformInMiddle), const_cast<char**>(platformInMiddle)).value());
}

TEST(ParsePlatformTest, InitResourcesWithQuotedMultipleOptions) {
    const char* platformWithQuotes[] = {
            "./vpux-opt",
            "--split-input-file",
            "--init-resources=\"someflag=aaa platform=NPU4000\"",
            "foo.mlir",
    };
    EXPECT_EQ(config::ArchKind::NPU40XX,
              vpux::parseParamsAndDeduceArch(std::size(platformWithQuotes), const_cast<char**>(platformWithQuotes))
                      .value());

    const char* platformBeforeOtherOptionsQuoted[] = {
            "./vpux-opt",
            "--split-input-file",
            "--init-resources=\"platform=NPU3720 resource-option=DefaultHW\"",
            "foo.mlir",
    };
    EXPECT_EQ(config::ArchKind::NPU37XX,
              vpux::parseParamsAndDeduceArch(std::size(platformBeforeOtherOptionsQuoted),
                                             const_cast<char**>(platformBeforeOtherOptionsQuoted))
                      .value());
}

TEST(ParseArchKindTest, InitCompilerWithMultipleOptions) {
    const char* archWithOtherOptions[] = {
            "./vpux-opt",
            "--split-input-file",
            "--init-compiler=someflag=aaa vpu-arch=NPU40XX",
            "foo.mlir",
    };
    EXPECT_EQ(config::ArchKind::NPU40XX,
              vpux::parseParamsAndDeduceArch(std::size(archWithOtherOptions), const_cast<char**>(archWithOtherOptions))
                      .value());

    const char* archInMiddle[] = {
            "./vpux-opt",
            "--split-input-file",
            "--init-compiler=flag1=value1 vpu-arch=NPU37XX flag2=value2",
            "foo.mlir",
    };
    EXPECT_EQ(config::ArchKind::NPU37XX,
              vpux::parseParamsAndDeduceArch(std::size(archInMiddle), const_cast<char**>(archInMiddle)).value());
}

TEST(ParseArchKindTest, InitCompilerWithQuotedMultipleOptions) {
    const char* archWithQuotes[] = {
            "./vpux-opt",
            "--split-input-file",
            "--init-compiler=\"someflag=aaa vpu-arch=NPU40XX\"",
            "foo.mlir",
    };
    EXPECT_EQ(config::ArchKind::NPU40XX,
              vpux::parseParamsAndDeduceArch(std::size(archWithQuotes), const_cast<char**>(archWithQuotes)).value());

    const char* archInMiddleQuoted[] = {
            "./vpux-opt",
            "--split-input-file",
            "--init-compiler=\"flag1=value1 vpu-arch=NPU37XX flag2=value2\"",
            "foo.mlir",
    };
    EXPECT_EQ(config::ArchKind::NPU37XX,
              vpux::parseParamsAndDeduceArch(std::size(archInMiddleQuoted), const_cast<char**>(archInMiddleQuoted))
                      .value());
}

TEST(ParseArchKindTest, InitResourcesWithMultipleOptions) {
    const char* archWithOtherOptions[] = {
            "./vpux-opt",
            "--split-input-file",
            "--init-resources=someflag=aaa vpu-arch=NPU40XX",
            "foo.mlir",
    };
    EXPECT_EQ(config::ArchKind::NPU40XX,
              vpux::parseParamsAndDeduceArch(std::size(archWithOtherOptions), const_cast<char**>(archWithOtherOptions))
                      .value());

    const char* archInMiddle[] = {
            "./vpux-opt",
            "--split-input-file",
            "--init-resources=flag1=value1 vpu-arch=NPU37XX flag2=value2",
            "foo.mlir",
    };
    EXPECT_EQ(config::ArchKind::NPU37XX,
              vpux::parseParamsAndDeduceArch(std::size(archInMiddle), const_cast<char**>(archInMiddle)).value());
}

TEST(ParseArchKindTest, InitResourcesWithQuotedMultipleOptions) {
    const char* archWithQuotes[] = {
            "./vpux-opt",
            "--split-input-file",
            "--init-resources=\"someflag=aaa vpu-arch=NPU40XX\"",
            "foo.mlir",
    };
    EXPECT_EQ(config::ArchKind::NPU40XX,
              vpux::parseParamsAndDeduceArch(std::size(archWithQuotes), const_cast<char**>(archWithQuotes)).value());

    const char* archInMiddleQuoted[] = {
            "./vpux-opt",
            "--split-input-file",
            "--init-resources=\"resource-opt=val vpu-arch=NPU37XX other-opt=val2\"",
            "foo.mlir",
    };
    EXPECT_EQ(config::ArchKind::NPU37XX,
              vpux::parseParamsAndDeduceArch(std::size(archInMiddleQuoted), const_cast<char**>(archInMiddleQuoted))
                      .value());
}
