"""Download the two benign command corpora used by the benchmark into research_data/raw (git-ignored).  Run: python -m research.fetch_data
  tldr-pages  : example commands with plain-language descriptions for thousands of command-line tools (CC BY 4.0)
  NL2Bash     : real shell one-liners written by programmers
No attack data is downloaded or used."""
import os
import urllib.request
import zipfile

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(HERE, "research_data", "raw")
TLDR = "https://github.com/tldr-pages/tldr/releases/latest/download/tldr.zip"
NL2BASH = "https://raw.githubusercontent.com/TellinaTool/nl2bash/master/data/bash/all.cm"


def main():
    os.makedirs(RAW, exist_ok=True)
    zp = os.path.join(RAW, "tldr.zip")
    if not os.path.exists(os.path.join(RAW, "tldr", "pages.en")):
        urllib.request.urlretrieve(TLDR, zp)
        with zipfile.ZipFile(zp) as z:
            z.extractall(os.path.join(RAW, "tldr"), members=[n for n in z.namelist() if n.split("/")[0] in ("pages", "pages.en")])
    nl = os.path.join(RAW, "nl2bash_all.cm")
    if not os.path.exists(nl):
        urllib.request.urlretrieve(NL2BASH, nl)
    print("data in", RAW)


if __name__ == "__main__":
    main()
