#!/usr/bin/env python
"""
Gera as sequências topológicas X_top (N, R, 18) de cada dataset, no formato lido por
ablacao_topologica.py.

Estrutura esperada:
    data/UCSB/Benign/*.png, data/UCSB/Malignant/*.png     (uma subpasta por classe)
    data/LA/1/, data/LA/6/, ...                           (idem CR, LG, NHL)
Saída:
    data/<DS>/topological.npz  com
        X             (N, R, 18) float32  medidas ACUMULADAS por raio, sem normalização
        y             (N,) int64          índice da classe
        classes       nomes das pastas, na ordem dos índices
        files         caminho relativo de cada imagem (alinhado a X e y)
        feature_names nomes das 18 colunas
    data/<DS>/.topo_cache_r{R}_ribas2024/  um .npy por imagem (permite retomar)

Por imagem, para cada raio r = 1..R e cada canal (R, G, B), a média e o desvio padrão
sobre os vértices de: grau de saída, força de saída e força de entrada (3 x 2 x 3 = 18).
A imagem é usada como está: uint8, resolução nativa, sem redimensionar nem normalizar.

Modelo (Ribas et al., 2024, Seção 3): aresta do pixel de menor para o de maior intensidade;
peso da aresta no raio r:  a = ((|p_i - p_j| + 1)(d + 1) - 1) / ((L + 1)(r + 1) - 1),
com d a distância euclidiana e L = 255. Força = soma dos pesos das arestas de saída / entrada.

Pré-requisito (compilar uma vez, a .so fica ao lado deste arquivo, em src/):
    g++ -O3 -shared -fPIC -fopenmp -o src/SSNModel.so src/SSNModel.cpp

Uso:
    uv run src/build_topological.py                       # todos os datasets
    uv run src/build_topological.py --datasets UCSB CR    # alguns
    uv run src/build_topological.py --rmax 15
    uv run src/build_topological.py --resize 256          # análise de resolução -> topological_256.npz

Com --resize N a imagem vira N x N por interpolação de área (sem preservar a proporção) e o
resultado vai para data/<DS>/topological_N.npz, com cache em .topo_cache_r{R}_ribas2024_N/.
A ordem de classes/arquivos é a mesma do topological.npz nativo, então y e files coincidem.
O C++ usa OpenMP: exporte OMP_NUM_THREADS=<nº de núcleos> ao rodar este script.
"""
import argparse
import ctypes
import re
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LIB_PATH = Path(__file__).resolve().parent / "SSNModel.so"
DATASETS = ["UCSB", "CR", "LA", "LG", "NHL"]
IMG_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
L = 255

CHANNELS = ["r", "g", "b"]
MEASURES = ["degree_out", "force_out", "force_in"]
# ordem das 18 colunas: canal -> medida -> (média, desvio padrão)
FEATURE_NAMES = [f"{m}_{ch}_{s}" for ch in CHANNELS for m in MEASURES for s in ("mean", "std")]


class NetworkMeasures(ctypes.Structure):
    _fields_ = [(n, ctypes.POINTER(ctypes.c_float)) for n in
                ("degree_out", "degree_in", "num_out", "num_in")]


class SSN:
    """Rede 'full' (intracanal + intercanal) e medidas por raio, via SSNModel.so."""

    def __init__(self, rmax=15, lib_path=LIB_PATH):
        if not Path(lib_path).exists():
            raise FileNotFoundError(
                f"{lib_path} não existe. Compile: "
                "g++ -O3 -shared -fPIC -fopenmp -o src/SSNModel.so src/SSNModel.cpp")
        self.rmax = rmax
        self.lib = ctypes.CDLL(str(lib_path))
        self.lib.model_full.argtypes = [ctypes.POINTER(ctypes.c_uint8),
                                        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
        self.lib.model_full.restype = ctypes.POINTER(NetworkMeasures)
        self.lib.free_network_measures.argtypes = [ctypes.POINTER(NetworkMeasures)]
        self.lib.free_network_measures.restype = None

    def features(self, img_rgb):
        """img_rgb: uint8 (H, W, 3) -> (rmax, 18) float64."""
        if img_rgb.ndim != 3 or img_rgb.shape[2] != 3 or img_rgb.dtype != np.uint8:
            raise ValueError("esperado uint8 (H, W, 3)")
        h, w, z = img_rgb.shape
        R = self.rmax
        img = np.ascontiguousarray(img_rgb)
        ptr = self.lib.model_full(img.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)), w, h, z, R)
        try:
            m = ptr.contents
            n = h * w * z * R

            def view(name):  # (H, W, C, R), sem cópia
                return np.ctypeslib.as_array(getattr(m, name), shape=(n,)).reshape(h, w, z, R)

            deg = view("degree_out")
            num_o, num_i = view("num_out"), view("num_in")
            for arr in (deg, num_o, num_i):  # acumula em r, in-place
                np.cumsum(arr, axis=3, out=arr)

            out = np.empty((R, 18), dtype=np.float64)
            for r in range(R):
                denom = (L + 1) * ((r + 1) + 1) - 1  # (L+1)(r+1)-1 com r = 1..R
                f_out = num_o[..., r] / denom
                f_in = num_i[..., r] / denom
                for c in range(z):
                    for k, v in enumerate((deg[:, :, c, r], f_out[:, :, c], f_in[:, :, c])):
                        out[r, c * 6 + 2 * k] = v.mean(dtype=np.float64)
                        out[r, c * 6 + 2 * k + 1] = v.std(dtype=np.float64)
            return out
        finally:
            self.lib.free_network_measures(ptr)


def natural_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def list_dataset(ds_dir):
    classes = sorted([d.name for d in ds_dir.iterdir() if d.is_dir() and not d.name.startswith(".")],
                     key=natural_key)
    items = []
    for ci, cname in enumerate(classes):
        files = sorted([p for p in (ds_dir / cname).iterdir()
                        if p.suffix.lower() in IMG_EXT], key=lambda p: natural_key(p.name))
        items += [(p, ci) for p in files]
    return classes, items


def build_dataset(ds, args, ssn):
    ds_dir = Path(args.data_dir) / ds
    classes, items = list_dataset(ds_dir)
    if args.limit:  # só para teste rápido
        seen, kept = {}, []
        for p, ci in items:
            seen[ci] = seen.get(ci, 0) + 1
            if seen[ci] <= args.limit:
                kept.append((p, ci))
        items = kept
    res_tag = f"_{args.resize}" if args.resize else ""   # "" = resolução nativa (compatível com o cache antigo)
    cache = ds_dir / f".topo_cache_r{args.rmax}_ribas2024{res_tag}"
    cache.mkdir(exist_ok=True)

    print(f"[{ds}] {len(items)} imagens, classes: {classes}", flush=True)
    X, y, files = [], [], []
    t0 = time.time()
    for k, (p, ci) in enumerate(items):
        rel = p.relative_to(ds_dir)
        cf = cache / (str(rel).replace("/", "__") + ".npy")
        if cf.exists():
            feat = np.load(cf)
        else:
            bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if bgr is None:
                raise IOError(f"não foi possível ler {p}")
            if args.resize:  # análise de sensibilidade: lado x lado, interpolação de área, sem preservar proporção
                bgr = cv2.resize(bgr, (args.resize, args.resize), interpolation=cv2.INTER_AREA)
            feat = ssn.features(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            np.save(cf, feat)
        X.append(feat)
        y.append(ci)
        files.append(str(rel))
        if (k + 1) % 10 == 0 or k + 1 == len(items):
            print(f"[{ds}] {k + 1}/{len(items)}  ({time.time() - t0:.0f}s)", flush=True)

    X = np.stack(X).astype(np.float32)
    out = ds_dir / (f"topological_{args.resize}.npz" if args.resize else "topological.npz")
    np.savez(out, X=X, y=np.array(y, dtype=np.int64), classes=np.array(classes),
             files=np.array(files), feature_names=np.array(FEATURE_NAMES),
             rmax=args.rmax, weight="ribas2024", resize=args.resize)
    print(f"[{ds}] salvo {out}  X{X.shape}  imagens por classe: "
          f"{np.bincount(y).tolist()}  ({time.time() - t0:.0f}s)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--data_dir", default=str(ROOT / "data"))
    ap.add_argument("--rmax", type=int, default=15)
    ap.add_argument("--limit", type=int, default=0, help="máx. de imagens por classe (teste)")
    ap.add_argument("--resize", type=int, default=0,
                    help="se > 0, redimensiona cada imagem para resize x resize (INTER_AREA) antes da rede "
                         "e salva topological_<resize>.npz; 0 = resolução nativa (topological.npz)")
    args = ap.parse_args()

    ssn = SSN(args.rmax)
    for ds in args.datasets:
        build_dataset(ds, args, ssn)


if __name__ == "__main__":
    main()
