"""Generate the code reference pages and navigation."""

from pathlib import Path
import os

import mkdocs_gen_files

nav = mkdocs_gen_files.Nav()

root = Path(__file__).parent.parent
src = root / "spark_instructor"

print(f"Root directory: {root}")
print(f"Source directory: {src}")
print(f"Current working directory: {os.getcwd()}")

if not src.exists():
    print(f"Error: The directory {src} does not exist.")
else:
    print(f"Contents of {src}:")
    for item in src.iterdir():
        print(f"  {item}")

for path in sorted(src.rglob("*.py")):
    print(f"Processing file: {path}")
    module_path = path.relative_to(src).with_suffix("")
    doc_path = path.relative_to(src).with_suffix(".md")
    full_doc_path = Path("reference", doc_path)

    parts = tuple(module_path.parts)

    if parts[-1] == "__init__":
        parts = parts[:-1]
        doc_path = doc_path.with_name("index.md")
        full_doc_path = full_doc_path.with_name("index.md")
    elif parts[-1] == "__main__":
        continue

    # Skip empty parts (root __init__.py)
    if not parts:
        continue

    nav[parts] = doc_path.as_posix()

    with mkdocs_gen_files.open(full_doc_path, "w") as fd:
        ident = ".".join(parts)
        fd.write(f"::: {ident}")

    mkdocs_gen_files.set_edit_path(full_doc_path, path.relative_to(root))

with mkdocs_gen_files.open("reference/SUMMARY.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())

print("Script completed.")
