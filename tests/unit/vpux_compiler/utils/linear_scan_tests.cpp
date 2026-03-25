//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/utils/linear_scan.hpp"

#include <gtest/gtest.h>

using namespace vpux;

// Run cmd: npuUnitTests --gtest_filter="MLIR_LinearScanTests.*"

class MLIR_LinearScanTests : public ::testing::Test {
protected:
    struct Handler;

    struct LiveRange {
        bool alive = true;
        bool fixed = false;
        AddressType size = 0;
        AddressType alignment = 1;
        AddressType addr = InvalidAddress;
        int spillWeight = 0;
        bool spilled = false;
    };

    struct Handler {
        bool isAlive(LiveRange* r) const {
            return r->alive;
        }

        bool isFixedAlloc(LiveRange* r) const {
            return r->fixed;
        }

        AddressType getSize(LiveRange* r) const {
            return r->size;
        }

        AddressType getAlignment(LiveRange* r) const {
            return r->alignment;
        }

        AddressType getAddress(LiveRange* r) const {
            return r->addr;
        }

        void allocated(LiveRange* r, AddressType addr) const {
            r->addr = addr;
        }

        void freed(LiveRange*) const {
        }

        int getSpillWeight(LiveRange* r) const {
            return r->spillWeight;
        }

        bool spilled(LiveRange* r) const {
            r->spilled = true;
            return true;
        }
    };
};

TEST_F(MLIR_LinearScanTests, Alloc) {
    // When memory is full, the range with lowest spill weight is evicted
    {
        LinearScan<LiveRange*, Handler> s(4);

        LiveRange r1;
        r1.size = 2;

        LiveRange r2;
        r2.size = 2;
        r2.spillWeight = -1;

        LiveRange r3;
        r3.size = 2;

        ASSERT_TRUE(s.alloc({&r1, &r2}));
        ASSERT_EQ(s.liveRanges(), (SmallVector<LiveRange*>{&r1, &r2}));

        ASSERT_TRUE(s.alloc({&r3}));
        ASSERT_EQ(s.liveRanges(), (SmallVector<LiveRange*>{&r1, &r3}));

        ASSERT_FALSE(r1.spilled);
        ASSERT_TRUE(r2.spilled);
        ASSERT_FALSE(r3.spilled);

        r1.alive = false;
        r2.alive = false;
        r3.alive = false;
        s.freeDeadRanges();

        ASSERT_EQ(r1.addr, 0);
        ASSERT_EQ(r2.addr, 2);
        ASSERT_EQ(r3.addr, 2);

        ASSERT_TRUE(s.liveRanges().empty());
        ASSERT_EQ(s.gaps().size(), 1);
    }

    // Allocation fails when there's not enough space and spilling is disabled
    {
        LinearScan<LiveRange*, Handler> s(4);

        LiveRange r1;
        r1.size = 2;

        LiveRange r2;
        r2.size = 2;
        r2.spillWeight = -1;

        LiveRange r3;
        r3.size = 2;

        ASSERT_TRUE(s.alloc({&r1, &r2}, false));
        ASSERT_EQ(s.liveRanges(), (SmallVector<LiveRange*>{&r1, &r2}));

        ASSERT_FALSE(s.alloc({&r3}, false));
        ASSERT_EQ(s.liveRanges(), (SmallVector<LiveRange*>{&r1, &r2}));

        ASSERT_FALSE(r1.spilled);
        ASSERT_FALSE(r2.spilled);

        r1.alive = false;
        r2.alive = false;
        r3.alive = false;
        s.freeDeadRanges();

        ASSERT_EQ(r1.addr, 0);
        ASSERT_EQ(r2.addr, 2);

        ASSERT_TRUE(s.liveRanges().empty());
    }

    // Tests that lower weight objects are spilled first when space is needed
    {
        LinearScan<LiveRange*, Handler> s(4);

        LiveRange r1;
        r1.size = 1;
        r1.spillWeight = 1;

        LiveRange r2;
        r2.size = 1;
        r2.spillWeight = 3;

        LiveRange r3;
        r3.size = 1;
        r3.spillWeight = 4;

        LiveRange r4;
        r4.size = 1;
        r4.spillWeight = 2;

        LiveRange r5;
        r5.size = 2;
        r5.spillWeight = 5;

        ASSERT_TRUE(s.alloc({&r1, &r2, &r3, &r4}));
        ASSERT_EQ(r1.addr, 0);
        ASSERT_EQ(r2.addr, 1);
        ASSERT_EQ(r3.addr, 2);
        ASSERT_EQ(r4.addr, 3);

        ASSERT_FALSE(r1.spilled);
        ASSERT_FALSE(r2.spilled);
        ASSERT_FALSE(r3.spilled);
        ASSERT_FALSE(r4.spilled);

        ASSERT_TRUE(s.alloc({&r5}));
        ASSERT_EQ(s.liveRanges().size(), 3);
        ASSERT_EQ(r1.addr, 0);
        ASSERT_EQ(r2.addr, 1);
        ASSERT_EQ(r3.addr, 2);
        ASSERT_EQ(r4.addr, 3);
        ASSERT_EQ(r5.addr, 0);

        ASSERT_TRUE(r1.spilled);
        ASSERT_TRUE(r2.spilled);
        ASSERT_FALSE(r3.spilled);
        ASSERT_FALSE(r4.spilled);
        ASSERT_FALSE(r5.spilled);

        r1.alive = false;
        r2.alive = false;
        r3.alive = false;
        r4.alive = false;
        r5.alive = false;
        s.freeDeadRanges();

        ASSERT_TRUE(s.liveRanges().empty());
    }

    // Tests spilling when there's only room for one object at a time
    {
        LinearScan<LiveRange*, Handler> s(1);

        LiveRange r1;
        r1.size = 1;

        LiveRange r2;
        r2.size = 1;

        ASSERT_FALSE(s.alloc({&r1, &r2}));

        ASSERT_FALSE(r1.spilled);
        ASSERT_FALSE(r2.spilled);

        ASSERT_EQ(r1.addr, InvalidAddress);
        ASSERT_EQ(r2.addr, InvalidAddress);

        ASSERT_TRUE(s.alloc({&r1}));

        ASSERT_FALSE(r1.spilled);
        ASSERT_FALSE(r2.spilled);

        ASSERT_EQ(r1.addr, 0);
        ASSERT_EQ(r2.addr, InvalidAddress);

        ASSERT_TRUE(s.alloc({&r2}));

        ASSERT_TRUE(r1.spilled);
        ASSERT_FALSE(r2.spilled);

        ASSERT_EQ(r1.addr, 0);
        ASSERT_EQ(r2.addr, 0);
    }
}

TEST_F(MLIR_LinearScanTests, AllocFixed) {
    LinearScan<LiveRange*, Handler> s(4);

    // Allocate 4 ranges of size 1, filling the entire memory [0-4)
    LiveRange r1;
    r1.size = 1;

    LiveRange r2;
    r2.size = 1;

    LiveRange r3;
    r3.size = 1;

    LiveRange r4;
    r4.size = 1;

    ASSERT_TRUE(s.alloc({&r1, &r2, &r3, &r4}));

    ASSERT_FALSE(r1.spilled);
    ASSERT_FALSE(r2.spilled);
    ASSERT_FALSE(r3.spilled);
    ASSERT_FALSE(r4.spilled);

    // r5 at fixed address 1 with size 2 [1-3)
    // Conflicts with r2 [1-2) and r3 [2-3)
    LiveRange r5;
    r5.size = 2;
    r5.addr = 1;
    r5.fixed = true;

    // Should fail without spilling - cannot fit without evicting r2 and r3
    ASSERT_FALSE(s.alloc({&r5}, false));
    ASSERT_FALSE(r1.spilled);
    ASSERT_FALSE(r2.spilled);
    ASSERT_FALSE(r3.spilled);
    ASSERT_FALSE(r4.spilled);

    // Should succeed with spilling - r2 and r3 are evicted to make room
    ASSERT_TRUE(s.alloc({&r5}, true));
    ASSERT_FALSE(r1.spilled);
    ASSERT_TRUE(r2.spilled);
    ASSERT_TRUE(r3.spilled);
    ASSERT_FALSE(r4.spilled);
    ASSERT_FALSE(r5.spilled);

    ASSERT_EQ(r5.addr, 1);
}

TEST_F(MLIR_LinearScanTests, RollbackOnFailure) {
    LinearScan<LiveRange*, Handler> s(4);

    LiveRange r1;
    r1.size = 2;

    LiveRange r2;
    r2.size = 3;  // This will fail - total size 5 > 4

    // Should fail and rollback r1's allocation
    ASSERT_FALSE(s.alloc({&r1, &r2}, false));

    // Verify r1 was not allocated (rollback worked)
    ASSERT_EQ(r1.addr, InvalidAddress);
    ASSERT_EQ(r2.addr, InvalidAddress);
    ASSERT_TRUE(s.liveRanges().empty());

    // Memory should be completely free
    ASSERT_EQ(s.totalFreeSize(), 4);
    ASSERT_EQ(s.gaps().size(), 1);

    // Now r1 should be able to allocate successfully
    ASSERT_TRUE(s.alloc({&r1}, false));
    ASSERT_EQ(r1.addr, 0);
}

TEST_F(MLIR_LinearScanTests, MixedFixedAndDynamicAlloc) {
    LinearScan<LiveRange*, Handler> s(100);

    LiveRange r1;
    r1.size = 10;
    r1.addr = 20;
    r1.fixed = true;

    LiveRange r2;
    r2.size = 10;  // Dynamic

    LiveRange r3;
    r3.size = 10;
    r3.addr = 50;
    r3.fixed = true;

    LiveRange r4;
    r4.size = 15;  // Dynamic

    // Mix fixed and dynamic in same allocation call
    ASSERT_TRUE(s.alloc({&r1, &r2, &r3, &r4}));

    // Fixed addresses should remain
    ASSERT_EQ(r1.addr, 20);
    ASSERT_EQ(r3.addr, 50);

    // Dynamic should not overlap with fixed
    ASSERT_TRUE((r2.addr + r2.size <= r1.addr) || (r2.addr >= r1.addr + r1.size));
    ASSERT_TRUE((r4.addr + r4.size <= r1.addr) || (r4.addr >= r1.addr + r1.size));
    ASSERT_TRUE((r4.addr + r4.size <= r3.addr) || (r4.addr >= r3.addr + r3.size));
}

TEST_F(MLIR_LinearScanTests, SpillAndReuse) {
    LinearScan<LiveRange*, Handler> s(10);

    LiveRange r1;
    r1.size = 10;
    r1.spillWeight = 1;

    ASSERT_TRUE(s.alloc({&r1}));
    ASSERT_EQ(r1.addr, 0);

    // This should spill r1
    LiveRange r2;
    r2.size = 10;
    r2.spillWeight = 2;

    ASSERT_TRUE(s.alloc({&r2}));
    ASSERT_TRUE(r1.spilled);
    ASSERT_FALSE(r2.spilled);
    ASSERT_EQ(r2.addr, 0);  // Reused r1's memory

    // Mark r2 as dead and free it
    r2.alive = false;
    s.freeDeadRanges();

    // Now allocate r1 again
    r1.spilled = false;
    ASSERT_TRUE(s.alloc({&r1}));
    ASSERT_EQ(r1.addr, 0);  // Memory was freed and reused
}

TEST_F(MLIR_LinearScanTests, CanAllocQuery) {
    LinearScan<LiveRange*, Handler> s(20);

    LiveRange r1;
    r1.size = 10;

    LiveRange r2;
    r2.size = 15;

    // Should be able to allocate r1
    ASSERT_TRUE(s.canAlloc({&r1}));

    // Actually allocate r1
    ASSERT_TRUE(s.alloc({&r1}));

    // Should not be able to allocate r2 (10 + 15 > 20)
    ASSERT_FALSE(s.canAlloc({&r2}));

    // canAlloc should not modify state
    ASSERT_EQ(s.liveRanges().size(), 1);
    ASSERT_EQ(s.totalFreeSize(), 10);
}

TEST_F(MLIR_LinearScanTests, AllocWithReservedSpace) {
    LinearScan<LiveRange*, Handler> s(100);

    LiveRange r1;
    r1.size = 20;

    LiveRange r2;
    r2.size = 30;

    // Reserve 40 bytes, then allocate r1 and r2
    auto reserveAddr = s.allocWithReservedSpace({&r1, &r2}, 40, 1, false);

    // Should succeed - 40 (reserve) + 20 + 30 = 90 < 100
    ASSERT_NE(reserveAddr, InvalidAddress);

    // Verify allocations
    ASSERT_EQ(s.liveRanges().size(), 2);

    // All should fit in the remaining 60 bytes
    ASSERT_TRUE(r1.addr != InvalidAddress);
    ASSERT_TRUE(r2.addr != InvalidAddress);
}

TEST_F(MLIR_LinearScanTests, LargeObjectAllocation) {
    LinearScan<LiveRange*, Handler> s(1024);

    LiveRange r1;
    r1.size = 512;
    r1.alignment = 64;

    LiveRange r2;
    r2.size = 600;  // Increased to ensure it doesn't fit
    r2.alignment = 64;

    ASSERT_TRUE(s.alloc({&r1}));
    ASSERT_EQ(r1.addr % 64, 0);

    // Should fail - not enough space (512 + 600 > 1024)
    ASSERT_FALSE(s.alloc({&r2}, false));

    // Free r1
    r1.alive = false;
    s.freeDeadRanges();

    // Now r2 should succeed
    ASSERT_TRUE(s.alloc({&r2}, false));
    ASSERT_EQ(r2.addr % 64, 0);
}

TEST_F(MLIR_LinearScanTests, MultipleSpillCandidates) {
    LinearScan<LiveRange*, Handler> s(10);

    LiveRange r1;
    r1.size = 3;
    r1.spillWeight = 1;

    LiveRange r2;
    r2.size = 3;
    r2.spillWeight = 2;

    LiveRange r3;
    r3.size = 4;
    r3.spillWeight = 3;

    ASSERT_TRUE(s.alloc({&r1, &r2, &r3}));

    // Allocate large buffer that requires spilling multiple
    LiveRange r4;
    r4.size = 7;
    r4.spillWeight = 10;

    ASSERT_TRUE(s.alloc({&r4}));

    // The algorithm spills candidates until enough space is available
    // To fit r4 (size 7), it needs to free at least 7 bytes
    // r1(3) + r2(3) = 6 bytes is not enough, so r3(4) must also be spilled
    ASSERT_TRUE(r1.spilled);   // weight 1 - spilled
    ASSERT_TRUE(r2.spilled);   // weight 2 - spilled
    ASSERT_TRUE(r3.spilled);   // weight 3 - also spilled to make room
    ASSERT_FALSE(r4.spilled);  // weight 10 - not spilled
}

TEST_F(MLIR_LinearScanTests, FixedAddressConflict) {
    LinearScan<LiveRange*, Handler> s(100);

    LiveRange r1;
    r1.size = 20;
    r1.addr = 10;
    r1.fixed = true;

    ASSERT_TRUE(s.alloc({&r1}));

    // Try to allocate another fixed at overlapping address
    LiveRange r2;
    r2.size = 20;
    r2.addr = 20;  // Overlaps with r1 (10-30)
    r2.fixed = true;

    // Should fail without spilling
    ASSERT_FALSE(s.alloc({&r2}, false));

    // Should succeed with spilling
    ASSERT_TRUE(s.alloc({&r2}, true));
    ASSERT_TRUE(r1.spilled);
    ASSERT_FALSE(r2.spilled);
}

TEST_F(MLIR_LinearScanTests, PartialOverlapSpilling) {
    LinearScan<LiveRange*, Handler> s(100);

    LiveRange r1;
    r1.size = 20;
    r1.spillWeight = 1;

    LiveRange r2;
    r2.size = 20;
    r2.spillWeight = 2;

    LiveRange r3;
    r3.size = 20;
    r3.spillWeight = 3;

    ASSERT_TRUE(s.alloc({&r1, &r2, &r3}));
    ASSERT_EQ(r1.addr, 0);
    ASSERT_EQ(r2.addr, 20);
    ASSERT_EQ(r3.addr, 40);

    // Fixed allocation that only conflicts with r2
    LiveRange r4;
    r4.size = 15;
    r4.addr = 25;  // Overlaps only with r2 (20-40)
    r4.fixed = true;

    ASSERT_TRUE(s.alloc({&r4}, true));

    // Only r2 should be spilled (the one that overlaps)
    ASSERT_FALSE(r1.spilled);
    ASSERT_TRUE(r2.spilled);
    ASSERT_FALSE(r3.spilled);
}

TEST_F(MLIR_LinearScanTests, RepeatedAllocFree) {
    LinearScan<LiveRange*, Handler> s(50);

    for (int i = 0; i < 10; ++i) {
        LiveRange r;
        r.size = 20;

        ASSERT_TRUE(s.alloc({&r}));
        ASSERT_NE(r.addr, InvalidAddress);

        r.alive = false;
        s.freeDeadRanges();

        // Should be back to initial state
        ASSERT_EQ(s.totalFreeSize(), 50);
        ASSERT_TRUE(s.liveRanges().empty());
    }
}

TEST_F(MLIR_LinearScanTests, DirectionDown) {
    LinearScan<LiveRange*, Handler> s(100);

    LiveRange r1;
    r1.size = 10;

    LiveRange r2;
    r2.size = 20;

    // Allocate from top down
    ASSERT_TRUE(s.alloc({&r1}, true, Partitioner::Direction::Down));
    ASSERT_EQ(r1.addr, 90);  // 100 - 10 = 90

    ASSERT_TRUE(s.alloc({&r2}, true, Partitioner::Direction::Down));
    ASSERT_EQ(r2.addr, 70);  // 90 - 20 = 70
}

TEST_F(MLIR_LinearScanTests, AlignmentRequirements) {
    LinearScan<LiveRange*, Handler> s(64);

    LiveRange r1;
    r1.size = 1;
    r1.alignment = 1;

    LiveRange r2;
    r2.size = 10;
    r2.alignment = 16;

    LiveRange r3;
    r3.size = 5;
    r3.alignment = 8;

    ASSERT_TRUE(s.alloc({&r1, &r2, &r3}));

    // r1 should be at 0
    ASSERT_EQ(r1.addr, 0);

    // r2 needs 16-byte alignment, should be at 16 (next aligned address after r1)
    ASSERT_EQ(r2.addr % 16, 0);

    // r3 needs 8-byte alignment
    ASSERT_EQ(r3.addr % 8, 0);
}

TEST_F(MLIR_LinearScanTests, EmptyAllocation) {
    LinearScan<LiveRange*, Handler> s(100);

    // Empty allocation should succeed
    SmallVector<LiveRange*> empty;
    ASSERT_TRUE(s.alloc(empty));

    ASSERT_TRUE(s.liveRanges().empty());
    ASSERT_EQ(s.totalFreeSize(), 100);
}

TEST_F(MLIR_LinearScanTests, AllocWithExcludedRegion) {
    // Test basic excluded region - allocation should avoid excluded area
    {
        LinearScan<LiveRange*, Handler> s(100);

        LiveRange r1;
        r1.size = 10;

        LiveRange r2;
        r2.size = 30;

        // Exclude region [20, 40)
        SmallVector<std::pair<AddressType, AddressType>> excluded = {{20, 20}};

        ASSERT_TRUE(s.allocWithExcludedRegion(excluded, {&r1, &r2}));

        // Both allocations should not overlap with [20, 40)
        ASSERT_TRUE((r1.addr + r1.size <= 20) || (r1.addr >= 40));
        ASSERT_TRUE((r2.addr + r2.size <= 20) || (r2.addr >= 40));

        // r1 should be at 0, [0-10)
        ASSERT_EQ(r1.addr, 0);
        // r2 should be at 40, [40-70)
        ASSERT_EQ(r2.addr, 40);
    }

    // Test multiple excluded regions
    {
        LinearScan<LiveRange*, Handler> s(100);

        LiveRange r1;
        r1.size = 10;

        LiveRange r2;
        r2.size = 30;

        LiveRange r3;
        r3.size = 10;

        // Exclude regions [0, 20) and [50, 70)
        SmallVector<std::pair<AddressType, AddressType>> excluded = {{0, 20}, {50, 20}};

        ASSERT_TRUE(s.allocWithExcludedRegion(excluded, {&r1, &r2, &r3}));

        // All allocations should not overlap with excluded regions
        ASSERT_TRUE(r1.addr >= 20 && (r1.addr + r1.size <= 50 || r1.addr >= 70));
        ASSERT_TRUE(r2.addr >= 20 && (r2.addr + r2.size <= 50 || r2.addr >= 70));
        ASSERT_TRUE(r3.addr >= 20 && (r3.addr + r3.size <= 50 || r3.addr >= 70));

        // r1 should be at 20, [20-30)
        ASSERT_EQ(r1.addr, 20);
        // r2 should be at 70, [70-100)
        ASSERT_EQ(r2.addr, 70);
        // r3 should be at 30, [30-40)
        ASSERT_EQ(r3.addr, 30);
    }

    // Test allocation with excluded region and spilling
    {
        LinearScan<LiveRange*, Handler> s(60);

        LiveRange r1;
        r1.size = 15;
        r1.spillWeight = 5;

        LiveRange r2;
        r2.size = 15;
        r2.spillWeight = 3;

        ASSERT_TRUE(s.alloc({&r1, &r2}));
        // r1 at [0, 15), r2 at [15, 30), total used: 30 bytes

        LiveRange r3;
        r3.size = 20;
        r3.spillWeight = 10;

        // Exclude region [40, 50) - doesn't overlap with r1 or r2
        SmallVector<std::pair<AddressType, AddressType>> excluded = {{40, 10}};

        // Available without spilling: [30, 40) = 10 bytes, [50, 60) = 10 bytes
        // Not enough for r3 (20 bytes), so r2 must be spilled
        ASSERT_TRUE(s.allocWithExcludedRegion(excluded, {&r3}, true));

        // r2 should be spilled (lower weight), giving [15, 40) = 25 bytes
        // r3 can fit in [15, 35)
        ASSERT_TRUE(r2.spilled);
        ASSERT_FALSE(r1.spilled);
        ASSERT_FALSE(r3.spilled);

        // r3 should be allocated and not overlap with excluded region [40, 50)
        ASSERT_TRUE((r3.addr + r3.size <= 40) || (r3.addr >= 50));
        ASSERT_EQ(r3.addr, 15);  // Should reuse r2's space
    }

    // Test allocation failure when space is insufficient with excluded region
    {
        LinearScan<LiveRange*, Handler> s(20);

        LiveRange r1;
        r1.size = 15;

        // Exclude region [0, 15) leaving only 5 bytes available
        SmallVector<std::pair<AddressType, AddressType>> excluded = {{0, 15}};

        // Try to allocate 15 bytes when only 5 are available - should fail without spilling
        ASSERT_FALSE(s.allocWithExcludedRegion(excluded, {&r1}, false));

        ASSERT_FALSE(r1.spilled);
        ASSERT_EQ(r1.addr, InvalidAddress);
    }

    // Test empty excluded region list
    {
        LinearScan<LiveRange*, Handler> s(100);

        LiveRange r1;
        r1.size = 10;

        SmallVector<std::pair<AddressType, AddressType>> excluded;

        ASSERT_TRUE(s.allocWithExcludedRegion(excluded, {&r1}));

        // Should allocate normally without any exclusions
        ASSERT_EQ(r1.addr, 0);
    }

    // Test with Direction::Down and excluded region
    {
        LinearScan<LiveRange*, Handler> s(100);

        LiveRange r1;
        r1.size = 10;

        // Exclude region [80, 100)
        SmallVector<std::pair<AddressType, AddressType>> excluded = {{80, 20}};

        ASSERT_TRUE(s.allocWithExcludedRegion(excluded, {&r1}, true, Partitioner::Direction::Down));

        // Should allocate from top, avoiding excluded region [80, 100)
        ASSERT_TRUE(r1.addr + r1.size <= 80);
        ASSERT_EQ(r1.addr, 70);  // 80 - 10 = 70
    }
}
