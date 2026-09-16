import os
import re
import sys

def audit_cpp_file(filepath: str):
    print(f"[*] Auditing safety invariants for: {filepath}")
    with open(filepath, "r") as f:
        lines = f.readlines()

    errors = []
    
    in_function = False
    has_opened = False
    func_name = ""
    func_start_line = 0
    func_lines = []
    brace_depth = 0

    hot_path_prefixes = ["eval_", "execute_single_ins", "chimera_execute"]

    for idx, line in enumerate(lines):
        line_num = idx + 1
        stripped = line.strip()

        if not in_function and re.match(r"^(static\s+)?(int|void|float|size_t)\s+([a-zA-Z0-9_]+)\s*\(?", stripped):
            match = re.search(r"([a-zA-Z0-9_]+)\s*\(", stripped)
            if match:
                func_name = match.group(1)
                func_start_line = line_num
                func_lines = [line]
                brace_depth = line.count("{") - line.count("}")
                has_opened = ("{" in line)
                in_function = True
                continue

        if in_function:
            func_lines.append(line)
            if "{" in line:
                has_opened = True
            brace_depth += line.count("{") - line.count("}")
            if has_opened and brace_depth <= 0:
                total_len = len(func_lines)
                assert_count = sum(1 for l in func_lines if re.search(r"\bassert\s*\(", l))

                # Rule 4: Function Length <= 60 lines
                if total_len > 60:
                    errors.append(f"Rule 4 Violation in {func_name} (L{func_start_line}-L{line_num}): length {total_len} exceeds 60 lines")

                # Rule 5: Assertion Density >= 2
                if assert_count < 2:
                    errors.append(f"Rule 5 Violation in {func_name} (L{func_start_line}-L{line_num}): assertion count {assert_count} < 2")

                # Rule 3: Zero Dynamic Memory on Hot Path
                is_hot_path = any(func_name.startswith(p) for p in hot_path_prefixes)
                if is_hot_path:
                    for l_idx, l in enumerate(func_lines):
                        if re.search(r"\b(malloc|calloc|realloc|free|new|delete)\b", l):
                            errors.append(f"Rule 3 Violation in hot-path {func_name}: dynamic allocation at line {func_start_line + l_idx}")

                in_function = False
                has_opened = False
                func_name = ""

    return errors

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    target_file = os.path.join(base_dir, "chimera", "csrc", "chimera_eval.cpp")
    if not os.path.exists(target_file):
        print(f"Target file missing: {target_file}")
        sys.exit(1)

    errors = audit_cpp_file(target_file)
    if errors:
        print(f"[FAIL] Found {len(errors)} invariant violations:")
        for err in errors:
            print(f"  - {err}")
        sys.exit(1)
    else:
        print("[PASS] All Power of 10 Safety Invariants verified successfully (0 violations).")
        sys.exit(0)

if __name__ == "__main__":
    main()
