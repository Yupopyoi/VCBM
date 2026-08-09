"""
Convert an RVC FAISS .index into a Unity TextAsset-compatible .bytes file.

The Unity side reproduces the RVC retrieval path with:
  - IVF coarse search, nprobe=1
  - k=8 neighbors
  - squared-L2 distance
  - weight = (1 / distance)^2, normalized

For a typical RVC v2 IndexIVFFlat this keeps the index logic very close
to WebUI while avoiding a FAISS/native runtime dependency inside Unity.

Usage:
    python export_rvc_index_unity.py ^
        --rvc-dir "...\Retrieval-based-Voice-Conversion-WebUI" ^
        --index "...\added_IVF..._v2.index" ^
        --output "...\rvc_retrieval_index.bytes"

If --index is omitted, the script searches <rvc-dir>/logs recursively.
If exactly one .index exists it is selected automatically; otherwise the
candidate list is printed and the script exits.
"""

import argparse
import os
import struct
import sys
from pathlib import Path

import numpy as np


MAGIC = b"RVCI"
VERSION = 1


def choose_index(rvc_dir: str, explicit: str | None) -> str:
    if explicit:
        path = os.path.abspath(explicit)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Index not found: {path}")
        return path

    logs = os.path.join(rvc_dir, "logs")
    candidates = []

    if os.path.isdir(logs):
        for root, _, files in os.walk(logs):
            for name in files:
                if name.lower().endswith(".index"):
                    candidates.append(
                        os.path.abspath(os.path.join(root, name))
                    )

    if len(candidates) == 1:
        print("[INFO] Auto-selected index:")
        print(candidates[0])
        return candidates[0]

    if not candidates:
        raise FileNotFoundError(
            f"No .index file was found under: {logs}"
        )

    print("[ERROR] Multiple .index files were found.")
    print("[ERROR] Drag the wanted .index onto the BAT, or pass --index.")
    print()
    for i, path in enumerate(candidates, 1):
        print(f"  [{i}] {path}")

    raise RuntimeError(
        "Index selection is ambiguous."
    )


def extract_ivf_data(index, faiss):
    """
    Return:
      centroids [nlist, d]
      grouped_vectors [ntotal, d]
      offsets [nlist+1]

    RVC normally uses IndexIVFFlat. We obtain all reconstructed vectors,
    then assign each vector to the same coarse quantizer used by IVF.
    """

    d = int(index.d)
    ntotal = int(index.ntotal)

    if ntotal <= 0:
        raise RuntimeError("FAISS index contains no vectors.")

    print(f"[INFO] FAISS type: {type(index).__name__}")
    print(f"[INFO] dimension : {d}")
    print(f"[INFO] ntotal    : {ntotal}")

    vectors = index.reconstruct_n(0, ntotal).astype(
        np.float32,
        copy=False,
    )

    # Typical RVC .index is IndexIVFFlat. extract_index_ivf also works
    # through several FAISS wrappers.
    try:
        ivf = faiss.extract_index_ivf(index)
    except Exception as exc:
        ivf = None
        print(
            "[WARN] Could not extract IVF structure; "
            f"falling back to one flat list: {exc}"
        )

    if ivf is None:
        nlist = 1
        centroids = np.zeros((1, d), dtype=np.float32)
        assignments = np.zeros(ntotal, dtype=np.int64)
    else:
        if int(ivf.metric_type) != int(faiss.METRIC_L2):
            raise RuntimeError(
                "This exporter currently supports L2 RVC indexes only. "
                f"metric_type={ivf.metric_type}"
            )

        nlist = int(ivf.nlist)

        quantizer = faiss.downcast_index(ivf.quantizer)

        centroids = quantizer.reconstruct_n(
            0,
            nlist,
        ).astype(np.float32, copy=False)

        # nprobe=1 list assignment: nearest coarse centroid.
        _, assignments_2d = quantizer.search(
            vectors,
            1,
        )
        assignments = assignments_2d[:, 0].astype(
            np.int64,
            copy=False,
        )

        print(f"[INFO] nlist     : {nlist}")
        print(f"[INFO] nprobe    : 1 (Unity export target)")

    counts = np.bincount(
        assignments,
        minlength=nlist,
    ).astype(np.int64)

    offsets = np.zeros(nlist + 1, dtype=np.int32)
    np.cumsum(
        counts,
        out=offsets[1:],
        dtype=np.int64,
    )

    grouped = np.empty_like(vectors)

    cursor = offsets[:-1].astype(np.int64).copy()

    for original_id in range(ntotal):
        list_id = int(assignments[original_id])
        dst = int(cursor[list_id])
        grouped[dst] = vectors[original_id]
        cursor[list_id] += 1

    return centroids, offsets, grouped, vectors, assignments


def weighted_retrieval_from_distances(
    db_vectors,
    ids,
    distances,
):
    valid = ids >= 0
    ids = ids[valid]
    distances = distances[valid]

    if len(ids) == 0:
        raise RuntimeError("No nearest neighbors returned.")

    d = np.maximum(
        distances.astype(np.float64),
        1e-12,
    )

    w = np.square(1.0 / d)
    w /= np.sum(w)

    return np.sum(
        db_vectors[ids] * w[:, None],
        axis=0,
    ).astype(np.float32)


def unity_style_query(
    q,
    centroids,
    offsets,
    grouped,
    k=8,
):
    centroid_dist = np.sum(
        (centroids - q[None, :]) ** 2,
        axis=1,
    )
    list_id = int(np.argmin(centroid_dist))

    begin = int(offsets[list_id])
    end = int(offsets[list_id + 1])

    if begin >= end:
        return q.copy()

    local = grouped[begin:end]
    dist = np.sum(
        (local - q[None, :]) ** 2,
        axis=1,
    )

    kk = min(k, len(local))
    local_ids = np.argpartition(
        dist,
        kk - 1,
    )[:kk]

    selected_dist = dist[local_ids]

    d = np.maximum(
        selected_dist.astype(np.float64),
        1e-12,
    )
    w = np.square(1.0 / d)
    w /= np.sum(w)

    return np.sum(
        local[local_ids] * w[:, None],
        axis=0,
    ).astype(np.float32)


def parity_test(
    index,
    original_vectors,
    centroids,
    offsets,
    grouped,
):
    """
    Compare against the actual FAISS index search used by RVC.
    For IndexIVFFlat with nprobe=1 this should be extremely close.
    """

    try:
        index.nprobe = 1
    except Exception:
        pass

    count = min(16, len(original_vectors))
    if count <= 0:
        return

    # Use slightly perturbed stored features so exact-zero self distance
    # doesn't dominate the validation.
    rng = np.random.default_rng(1234)
    queries = (
        original_vectors[:count]
        + rng.normal(
            0.0,
            1e-3,
            size=original_vectors[:count].shape,
        ).astype(np.float32)
    )

    score, ix = index.search(
        queries.astype(np.float32),
        8,
    )

    diffs = []

    for i, q in enumerate(queries):
        faiss_out = weighted_retrieval_from_distances(
            original_vectors,
            ix[i],
            score[i],
        )

        unity_out = unity_style_query(
            q,
            centroids,
            offsets,
            grouped,
            k=8,
        )

        diffs.append(
            np.max(np.abs(faiss_out - unity_out))
        )

    max_diff = float(np.max(diffs))
    mean_diff = float(np.mean(diffs))

    print(
        f"[INFO] FAISS/Unity retrieval parity: "
        f"max_abs={max_diff:.8g}, mean_max_abs={mean_diff:.8g}"
    )

    # Different FAISS wrappers/training details can cause small differences.
    # We report rather than hard fail; the exported database remains valid.


def write_bytes(
    path: str,
    centroids: np.ndarray,
    offsets: np.ndarray,
    grouped: np.ndarray,
):
    nlist, d = centroids.shape
    ntotal = grouped.shape[0]

    os.makedirs(
        os.path.dirname(os.path.abspath(path)),
        exist_ok=True,
    )

    with open(path, "wb") as f:
        f.write(MAGIC)
        f.write(
            struct.pack(
                "<iiii",
                VERSION,
                int(d),
                int(nlist),
                int(ntotal),
            )
        )

        f.write(
            np.asarray(
                centroids,
                dtype="<f4",
                order="C",
            ).tobytes(order="C")
        )

        f.write(
            np.asarray(
                offsets,
                dtype="<i4",
                order="C",
            ).tobytes(order="C")
        )

        f.write(
            np.asarray(
                grouped,
                dtype="<f4",
                order="C",
            ).tobytes(order="C")
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rvc-dir",
        required=True,
    )
    parser.add_argument(
        "--index",
        default=None,
    )
    parser.add_argument(
        "--output",
        required=True,
    )
    args = parser.parse_args()

    rvc_dir = os.path.abspath(args.rvc_dir)

    if not os.path.isdir(rvc_dir):
        raise FileNotFoundError(
            f"RVC directory not found: {rvc_dir}"
        )

    try:
        import faiss
    except ImportError:
        print("[ERROR] faiss is not installed in this Python environment.")
        print("[ERROR] Use the same RVC .venv that WebUI uses.")
        raise

    index_path = choose_index(
        rvc_dir,
        args.index,
    )

    output = os.path.abspath(args.output)

    print()
    print("[INFO] Reading FAISS index:")
    print(index_path)

    index = faiss.read_index(index_path)

    centroids, offsets, grouped, original, assignments = (
        extract_ivf_data(index, faiss)
    )

    parity_test(
        index,
        original,
        centroids,
        offsets,
        grouped,
    )

    write_bytes(
        output,
        centroids,
        offsets,
        grouped,
    )

    size = os.path.getsize(output)

    print()
    print("[OK] Unity retrieval index was created:")
    print(output)
    print(f"[INFO] Size: {size:,} bytes")
    print(
        f"[INFO] Shape: nlist={centroids.shape[0]}, "
        f"vectors={grouped.shape[0]}, dim={grouped.shape[1]}"
    )
    print()
    print("[INFO] Put the .bytes file under Unity Assets/")
    print("[INFO] and assign it to Retrieval Index Asset.")


if __name__ == "__main__":
    main()