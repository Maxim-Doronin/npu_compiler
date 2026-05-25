#
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

#
# Dependency checker
#
# Validates header files included in the source and public header folders match the respective
# CMake target dependencies based on CMake target information collected during CMake
# configuration by dep_check.cmake and stored in a CSV file.
#

import re
import csv
import sys
import argparse
from pathlib import Path

# Read only the first N bytes of the source file for speed
FILE_HEADER_SIZE = 8192
DEBUG = False

INCLUDE_PATTERN = re.compile(r'#include\s+["<]([^">]+)[">]')


def debug(*args, **kwargs):
    print("[DEBUG]", *args, **kwargs) if DEBUG else None


def get_target_from_inc(include_path, prefix_target_map, sorted_keys):
    """
    Finds the target by matching the longest possible 'Prefix/TargetDir'
    combination found in the include path.
    """
    for key in sorted_keys:
        # We append a '/' to ensure we match a full directory name,
        # avoiding 'my/lib' matching 'my/library_extension/header.h'
        if include_path.startswith(key + "/"):
            return prefix_target_map[key]
    return None


class DependencyChecker:
    def __init__(self, root, prefix_target_map, sorted_keys):
        self.root = root
        self.prefix_target_map = prefix_target_map
        self.sorted_keys = sorted_keys
        self.referenced_deps = set()
        self.overall_success = True

    def check_dir(self, target_name, cmake_file, directory, allowed_deps, context_string):
        debug(f"  Scanning '{directory}'")
        if not directory.exists():
            return

        for file_path in directory.rglob('*'):
            if file_path.suffix not in ['.h', '.hpp', '.cpp', '.cc', '.cxx']:
                continue

            display_path = file_path.relative_to(self.root)
            debug(f"    Checking {display_path} for {context_string}")

            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    lines = f.read(FILE_HEADER_SIZE).splitlines()
            except Exception as e:
                print(f"error: failed to read '{file_path}': {e}")
                self.overall_success = False
                continue

            for line_num, line in enumerate(lines, 1):
                if line.strip().startswith("//"):  # Basic check to ignore comments at start of line
                    continue

                found_includes = INCLUDE_PATTERN.findall(line)
                for inc in found_includes:
                    inc_target = get_target_from_inc(inc, self.prefix_target_map, self.sorted_keys)
                    # Only validate if the include belongs to a known internal target
                    # and it is not a self-include
                    if inc_target and inc_target != target_name:
                        debug(f"      Found include '{inc}' from '{inc_target}'")
                        self.referenced_deps.add(inc_target)
                        if inc_target not in allowed_deps:
                            # GCC Format: file:line: error: message
                            print(f"{display_path}:{line_num}: error: "
                                  f"Illegal include '{inc}'. '{target_name}': target '{inc_target}' is not "
                                  f"in {context_string}. Check {cmake_file}")
                            self.overall_success = False


def validate_includes():
    parser = argparse.ArgumentParser(description="Validate C++ includes against CSV manifest.")
    parser.add_argument("root_dir", help="Root directory of the project")
    parser.add_argument("csv_file", help="Path to the CSV manifest")
    args = parser.parse_args()

    root = Path(args.root_dir)
    targets_info = {}
    prefix_target_map = {}

    # 1. Load dependency file (CSV)
    try:
        with open(args.csv_file, mode='r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            for row in reader:
                if not row:
                    continue
                row = [field.strip() for field in row]

                target, prefix, target_path, src_dir, inc_dir, pub_deps, all_deps, *_ = row

                pub_set = set(d.strip() for d in pub_deps.split(';') if d.strip())
                all_set = set(d.strip() for d in all_deps.split(';') if d.strip())

                targets_info[target] = {
                    'prefix': prefix, 'target_path': target_path,
                    'src_dir': src_dir, 'inc_dir': inc_dir,
                    'pub_deps': pub_set, 'all_deps': all_set
                }

                prefix_target_map[f"{prefix}/{target_path}"] = target
    except Exception as e:
        print(f"fatal error: failed to read dependency file: {e}")
        sys.exit(2)

    # Pre-sort prefixes by length (descending) for the greedy matcher
    sorted_keys = sorted(prefix_target_map.keys(), key=len, reverse=True)
    checker = DependencyChecker(root, prefix_target_map, sorted_keys)

    # 2. Process Targets
    for target_name, info in targets_info.items():
        debug(f"Checking target '{target_name}'")
        cmake_file = f"{info['src_dir']}/{info['target_path']}/CMakeLists.txt"
        checker.referenced_deps = set()

        # Paths to scan
        src_path = root / info['src_dir'] / info['target_path']
        inc_path = root / info['inc_dir'] / info['prefix'] / info['target_path']
        if not src_path.is_dir():
            print(f"error: no source directory '{src_path}' found for target '{target_name}'.")
            checker.overall_success = False
        if not inc_path.is_dir():
            print(f"warning: no include directory '{inc_path}' found for target '{target_name}'. Check {cmake_file}")

        # Rule: Source files check against Private Dependencies
        checker.check_dir(target_name, cmake_file, src_path, info['all_deps'], "dependencies")
        # Rule: Public Headers check against Public Dependencies
        checker.check_dir(target_name, cmake_file, inc_path, info['pub_deps'], "PUBLIC dependencies")

        unref_deps = info['all_deps'] - checker.referenced_deps
        for dep in unref_deps:
            if dep in targets_info.keys():  # Only report if the unreferenced dependency is a known target:
                print(f"error: Target '{target_name}' has an unreferenced dependency '{dep}'. Check {cmake_file}")
                checker.overall_success = False

    if not checker.overall_success:
        sys.exit(1)


if __name__ == "__main__":
    validate_includes()
