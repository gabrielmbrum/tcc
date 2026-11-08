#!/usr/bin/env python
"""
Encaminhador de compatibilidade para src/ablacao_via_visual.py.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ablacao_via_visual

if __name__ == "__main__":
    # Suporte a alias --arch -> --model para compatibilidade
    argv = []
    for arg in sys.argv:
        if arg == "--arch":
            argv.append("--model")
        elif arg.startswith("--arch="):
            argv.append(arg.replace("--arch=", "--model="))
        else:
            argv.append(arg)
    sys.argv = argv
    ablacao_via_visual.main()
