from pathlib import Path
import re

ROOT=Path(__file__).resolve().parents[1]

def test_no_external_runtime_imports_or_absolute_paths():
 forbidden=("src."+"n"+"g"+"a","experiments."+"neural"+"_galois","artifacts"+"/"+"neural"+"_galois","C:"+"\\\\Users")
 for path in ROOT.rglob("*.py"):
  text=path.read_text(encoding="utf-8")
  assert not any(value in text for value in forbidden),(path,[v for v in forbidden if v in text])

def test_no_generated_outputs_or_vendored_premap2():
 allowed={ROOT/"results/.gitkeep"}
 assert all(path in allowed for path in (ROOT/"results").rglob("*") if path.is_file())
 external=ROOT/"external/PREMAP2";assert not any(path.name==".git" for path in external.rglob("*"))
