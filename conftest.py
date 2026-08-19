import sys, os
_root = os.path.dirname(__file__)
sys.path.insert(0, _root)
# scripts/ too, so tests can import build_site the same way they import src.
sys.path.insert(0, os.path.join(_root, "scripts"))
