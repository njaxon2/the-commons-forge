#!/usr/bin/env python3
"""Extract V&V requirement traceability data from annotated test files.

Scans all test_*.py files in the given directory, extracts:
- Requirement IDs (R-XX-NN pattern)
- SHALL statements from class docstrings
- Test file and class/method mapping
- Verification tier classification

Outputs a CSV suitable for importing into the V&V traceability spreadsheet.
"""
import ast
import csv
import os
import re
import sys
from pathlib import Path


def extract_requirements(test_dir):
    """Extract requirements from all test files in the directory."""
    rows = []

    for test_file in sorted(Path(test_dir).glob("test_*.py")):
        try:
            tree = ast.parse(test_file.read_text())
        except SyntaxError:
            print(f"WARNING: Could not parse {test_file.name}", file=sys.stderr)
            continue

        filename = test_file.name

        # Determine verification tier
        if "integration" in filename or "e2e" in filename:
            tier = "Tier 3 (visual integration)"
        elif "gui" in filename:
            tier = "Tier 2 (widget / headless Qt)"
        else:
            tier = "Tier 1 (headless unit)"

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                class_name = node.name
                class_doc = ast.get_docstring(node) or ""

                # Extract requirement ID from class docstring
                req_match = re.search(r'(R-?[A-Z0-9]+-?\d+|R\d+)', class_doc)
                if not req_match:
                    continue

                req_id = req_match.group(1)

                # Extract SHALL statement
                shall_match = re.search(r'SHALL\s+(.+?)(?:\n\n|\n\s*Model|$)', class_doc, re.DOTALL)
                shall_text = shall_match.group(0).strip() if shall_match else ""
                # Clean up
                shall_text = re.sub(r'\s+', ' ', shall_text)[:200]

                # Extract model-user argument summary (first sentence)
                mu_match = re.search(r'Model-user argument[:\s]*\n?\s*(.+?)(?:\n\n|\n\s*Decomposition|$)', class_doc, re.DOTALL)
                mu_text = mu_match.group(1).strip() if mu_match else ""
                mu_text = re.sub(r'\s+', ' ', mu_text)[:300]

                # Count test methods in this class
                test_methods = [n.name for n in ast.walk(node)
                               if isinstance(n, ast.FunctionDef) and n.name.startswith("test_")]

                for method_name in test_methods:
                    # Try to get sub-requirement from method docstring
                    for m_node in ast.walk(node):
                        if isinstance(m_node, ast.FunctionDef) and m_node.name == method_name:
                            m_doc = ast.get_docstring(m_node) or ""
                            sub_match = re.search(r'(R-?[A-Z0-9]+-?\d+[\.\-]?\w*)', m_doc)
                            sub_id = sub_match.group(1) if sub_match else req_id
                            break
                    else:
                        sub_id = req_id

                    rows.append({
                        "Requirement ID": sub_id,
                        "Parent Requirement": req_id,
                        "SHALL Statement (summary)": shall_text[:200],
                        "Model-User Argument (summary)": mu_text[:200],
                        "Test File": filename,
                        "Test Class": class_name,
                        "Test Method": method_name,
                        "Verification Tier": tier,
                    })

            # Also handle standalone test functions (not in classes)
            elif isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
                if not any(isinstance(p, ast.ClassDef) for p in ast.walk(tree)):
                    pass  # Handled at module level
                func_doc = ast.get_docstring(node) or ""
                req_match = re.search(r'(R-?[A-Z0-9]+-?\d+[\.\-]?\w*)', func_doc)
                if req_match:
                    rows.append({
                        "Requirement ID": req_match.group(1),
                        "Parent Requirement": req_match.group(1).split(".")[0].split("-")[0] + "-" + "-".join(req_match.group(1).split("-")[1:3]) if "-" in req_match.group(1) else req_match.group(1),
                        "SHALL Statement (summary)": "",
                        "Model-User Argument (summary)": "",
                        "Test File": filename,
                        "Test Class": "(module-level)",
                        "Test Method": node.name,
                        "Verification Tier": tier,
                    })

    return rows


def main():
    if len(sys.argv) < 2:
        test_dir = os.path.expanduser("~/forge/tests")
    else:
        test_dir = sys.argv[1]

    rows = extract_requirements(test_dir)

    if not rows:
        print("No requirements found!", file=sys.stderr)
        sys.exit(1)

    # Write CSV
    output_file = os.path.join(os.path.dirname(test_dir), "vv_requirements_extracted.csv")
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "Requirement ID", "Parent Requirement", "SHALL Statement (summary)",
            "Model-User Argument (summary)", "Test File", "Test Class",
            "Test Method", "Verification Tier"
        ])
        writer.writeheader()
        writer.writerows(rows)

    # Summary
    unique_parents = set(r["Parent Requirement"] for r in rows)
    unique_reqs = set(r["Requirement ID"] for r in rows)
    unique_files = set(r["Test File"] for r in rows)

    print(f"Extracted {len(rows)} test-requirement mappings:")
    print(f"  {len(unique_parents)} parent requirements")
    print(f"  {len(unique_reqs)} unique requirement IDs")
    print(f"  {len(unique_files)} test files")
    print(f"  Output: {output_file}")


if __name__ == "__main__":
    main()
