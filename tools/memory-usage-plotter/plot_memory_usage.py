#!/usr/bin/env python3
#
# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
#
import re
import sys
import matplotlib.pyplot as plt


def parse_log_file(file_path):
    max_allocated_sizes = []
    max_allocated_sizes_increases = []
    fragmentation_max_allocated_sizes = []
    used_memories = []
    free_memories = []

    with open(file_path, 'r') as file:
        is_static_allocation_pass = False
        increase_due_to_fragmentation = False
        timestamp = 0
        for line in file:
            start_static_allocation_pass_match = re.search(r'Start Pass StaticAllocation', line)
            if start_static_allocation_pass_match:
                is_static_allocation_pass = True

            if not is_static_allocation_pass:
                continue

            end_static_allocation_pass_match = re.search(r'End Pass StaticAllocation', line)
            if end_static_allocation_pass_match:
                break

            max_allocated_match = re.search(r'Max allocated size (\d+)', line)
            if max_allocated_match:
                max_allocated_size = int(max_allocated_match.group(1))
                max_allocated_pair = (timestamp, max_allocated_size)
                if max_allocated_sizes and max_allocated_size > max_allocated_sizes[-1][1]:
                    max_allocated_sizes_increases.append(max_allocated_pair)
                max_allocated_sizes.append(max_allocated_pair)
                if increase_due_to_fragmentation:
                    fragmentation_max_allocated_sizes.append(max_allocated_pair)
                    increase_due_to_fragmentation = False
                continue

            used_memory_match = re.search(r'DDR used memory\s+(\d+)', line)
            if used_memory_match:
                used_memory = int(used_memory_match.group(1))
                used_memory_pair = (timestamp, used_memory)
                used_memories.append(used_memory_pair)
                continue

            free_memory_match = re.search(r'DDR free memory\s+(\d+)', line)
            if free_memory_match:
                free_memory = int(free_memory_match.group(1))
                free_memory_pair = (timestamp, free_memory)
                free_memories.append(free_memory_pair)
                timestamp += 1
                continue

            fragmentation_match = re.search(r'Increased allocation size due to fragmentation!', line)
            if fragmentation_match:
                increase_due_to_fragmentation = True

    return max_allocated_sizes, max_allocated_sizes_increases, fragmentation_max_allocated_sizes, used_memories, free_memories


def get_scale(max_allocated_size):
    byte = 1
    kilobyte = byte * 1024
    megabyte = kilobyte * 1024
    gigabyte = megabyte * 1024

    return ((gigabyte, "GB") if max_allocated_size >= gigabyte else
            (megabyte, "MB") if max_allocated_size >= megabyte else
            (kilobyte, "KB") if max_allocated_size >= kilobyte else
            (byte, "B"))


def convert_bytes_to_readable_size(sizes_in_bytes, divider):
    return [(timestamp, size / divider) for timestamp, size in sizes_in_bytes]


def plot_memory_usage(max_allocated_sizes, max_allocated_sizes_increases, fragmentation_max_allocated_sizes, used_memories, free_memories, out_file_path):
    max_allocated_size = max_allocated_sizes[-1][1]
    divider, unit = get_scale(max_allocated_size)

    max_allocated_sizes = convert_bytes_to_readable_size(max_allocated_sizes, divider)
    max_allocated_sizes_increases = convert_bytes_to_readable_size(max_allocated_sizes_increases, divider)
    fragmentation_max_allocated_sizes = convert_bytes_to_readable_size(fragmentation_max_allocated_sizes, divider)
    used_memories = convert_bytes_to_readable_size(used_memories, divider)
    free_memories = convert_bytes_to_readable_size(free_memories, divider)

    plt.figure(figsize=(40, 6))
    plt.scatter(*zip(*max_allocated_sizes_increases), label='Max Allocated Size\nIncreased',
                s=5, marker='o', color='magenta', zorder=2)
    if fragmentation_max_allocated_sizes:
        plt.scatter(*zip(*fragmentation_max_allocated_sizes),
                    label='Max Allocated Size\nIncreased Due To\nFragmentation', s=5, marker='^', color='red', zorder=3)
    plt.plot(*zip(*max_allocated_sizes), label='Max Allocated Size', color='blue', zorder=1)
    plt.plot(*zip(*used_memories), label='Used Memory', color='red', zorder=0)
    plt.plot(*zip(*free_memories), label='Free Memory', color='green', zorder=0)

    plt.xlabel('Timestamp')
    plt.xlim(xmin=0)
    plt.ylabel(f'Memory Size [{unit}]')
    plt.ylim(ymin=0)
    plt.title('Memory Usage Over Time')
    plt.legend()
    plt.xticks(rotation=45)
    plt.yticks()
    plt.tight_layout()
    plt.ticklabel_format(style='plain')
    plt.savefig(out_file_path)


if __name__ == "__main__":
    try:
        in_file = sys.argv[1]
        out_file = sys.argv[2]
    except Exception:
        print(f"Usage: {sys.argv[0]} INPUT_LOG OUTPUT_PNG")
        exit(1)

    max_allocated_sizes, max_allocated_sizes_increases, fragmentation_max_allocated_sizes, used_memories, free_memories = parse_log_file(
        in_file)
    plot_memory_usage(max_allocated_sizes, max_allocated_sizes_increases,
                      fragmentation_max_allocated_sizes, used_memories, free_memories, out_file)
