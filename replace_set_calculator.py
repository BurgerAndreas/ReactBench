#!/usr/bin/env python3
"""
Script to recursively replace .set_calculator(...) with .calc = ... in all files.
Use --dry-run to preview changes without modifying files.
"""

import os
import re
import argparse
from pathlib import Path


def is_text_file(filepath):
    """Check if file is likely a text file by trying to read it."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            f.read(1024)  # Try to read first 1KB
        return True
    except (UnicodeDecodeError, PermissionError):
        return False


def replace_set_calculator(content):
    """Replace .set_calculator(...) with .calc = ... using regex."""
    # Pattern: .set_calculator(anything_inside_parentheses)
    pattern = r"\.set_calculator\(([^)]*)\)"
    replacement = r".calc = \1"

    lines = content.split("\n")
    changes = []
    new_lines = []

    for line_num, line in enumerate(lines, 1):
        old_line = line
        new_line = re.sub(pattern, replacement, line)
        new_lines.append(new_line)

        if old_line != new_line:
            changes.append((line_num, old_line, new_line))

    new_content = "\n".join(new_lines)
    return new_content, changes


def process_file(filepath, dry_run=False):
    """Process a single file, optionally in dry-run mode."""
    if not is_text_file(filepath):
        return False, "Not a text file", []

    try:
        # Read original content
        with open(filepath, "r", encoding="utf-8") as f:
            original_content = f.read()

        # Apply replacement
        new_content, changes = replace_set_calculator(original_content)

        # Check if any changes were made
        if not changes:
            return False, "No changes needed", []

        if dry_run:
            return True, f"Would modify {len(changes)} line(s)", changes
        else:
            # Write modified content
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(new_content)

            return True, f"Modified {len(changes)} line(s)", changes

    except Exception as e:
        return False, f"Error: {e}", []


def main():
    """Main function to process all files recursively."""
    parser = argparse.ArgumentParser(
        description="Recursively replace .set_calculator(...) with .calc = ... in Python files"
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually apply changes (default is dry-run mode)",
    )
    args = parser.parse_args()

    start_dir = Path(".")
    dry_run = not args.apply

    mode = "Preview mode" if dry_run else "Replacement mode"
    print(
        f"{mode}: Recursively replacing .set_calculator(...) with .calc = ... in Python files"
    )
    print(f"Starting from: {start_dir.absolute()}")
    print("-" * 60)

    processed_count = 0
    modified_count = 0

    # Walk through all files recursively
    for root, dirs, files in os.walk(start_dir):
        for file in files:
            # Only process Python files
            if not file.endswith(".py"):
                continue

            filepath = Path(root) / file

            try:
                success, message, changes = process_file(filepath, dry_run=dry_run)
                if success:
                    modified_count += 1
                    print(f"✓ {filepath}: {message}")

                    # Show changed lines
                    if changes and (dry_run or len(changes) <= 5):  # Limit output
                        for line_num, old_line, new_line in changes:
                            print(f"  Line {line_num}:")
                            print(f"    - {old_line.strip()}")
                            print(f"    + {new_line.strip()}")
                else:
                    # Only show files that had errors
                    if "Error:" in message:
                        print(f"✗ {filepath}: {message}")

                processed_count += 1

            except Exception as e:
                print(f"✗ {filepath}: Unexpected error: {e}")

    print("-" * 60)
    print(f"Processed {processed_count} Python files")
    if dry_run:
        print(f"Would modify {modified_count} files")
        print("\nTo actually apply changes, run with: --apply")
    else:
        print(f"Modified {modified_count} files")


if __name__ == "__main__":
    main()
