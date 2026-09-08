"""Portable full project ZIP plus independently compilable arXiv source tarball."""
from pathlib import Path
import hashlib,json,tarfile,zipfile
ROOT=Path(__file__).resolve().parents[1]
skip={".aux",".log",".out",".blg",".fls",".fdb_latexmk",".pyc",".toc"}
paths=sorted(p for p in ROOT.rglob("*") if p.is_file() and "__pycache__" not in p.parts
             and p.suffix not in skip and p.name!="MANIFEST.json")
manifest={"algorithm":"sha256","scope":"All packaged files except MANIFEST.json",
          "files":{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
(ROOT/"MANIFEST.json").write_text(json.dumps(manifest,indent=2))
target=ROOT.parent/"M2_整合版_完整论文与代码.zip"
with zipfile.ZipFile(target,"w",zipfile.ZIP_DEFLATED) as archive:
    for path in paths+[ROOT/"MANIFEST.json"]:
        archive.write(path,"m2_integrated/"+str(path.relative_to(ROOT)))
with zipfile.ZipFile(target) as archive:
    assert archive.testzip() is None
source=ROOT.parent/"M2_整合版_arxiv_source.tar.gz"
with tarfile.open(source,"w:gz") as archive:
    for path in sorted((ROOT/"paper").rglob("*")):
        if path.is_file() and path.suffix in {".tex",".bib",".bbl",".pdf"} and path.name!="main.pdf":
            archive.add(path,arcname=str(path.relative_to(ROOT/"paper")))
    archive.add(ROOT/"docs/ARXIV_README.txt",arcname="README.txt")
print(f"{target}: {target.stat().st_size} bytes; {len(paths)+1} files")
print(f"{source}: {source.stat().st_size} bytes")
