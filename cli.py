#!/usr/bin/env python3
"""
Ruscomp command-line interface.

Features:
  * Compress/decompress text files with Huffman or RLE algorithms.
  * Displays compression ratio and throughput statistics.
  * Supports batch compression via multiple input files and an output directory.
  * Verbose and quiet logging modes.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import ruscomp


@dataclass
class CompressionResult:
    input_path: Path
    payload_path: Path
    metadata_path: Path
    original_bytes: int
    compressed_bytes: int
    duration: float

    @property
    def compression_ratio(self) -> float:
        if self.original_bytes == 0:
            return 1.0
        return self.compressed_bytes / float(self.original_bytes)

    @property
    def speed_mbps(self) -> float:
        if self.duration == 0:
            return float("inf")
        return (self.original_bytes / (1024 * 1024)) / self.duration


def configure_logging(args: argparse.Namespace) -> None:
    if args.verbose and args.quiet:
        raise SystemExit("Cannot use --verbose and --quiet simultaneously.")
    level = logging.INFO
    if args.verbose:
        level = logging.DEBUG
    elif args.quiet:
        level = logging.WARNING

    logging.basicConfig(
        level=level,
        format="%(levelname)s: %(message)s",
    )


def ensure_output_directory(path: Path) -> None:
    if not path.exists():
        logging.debug("Creating output directory %s", path)
        path.mkdir(parents=True, exist_ok=True)
    elif not path.is_dir():
        raise SystemExit(f"Output path {path} is not a directory.")


def compress_file(
    input_path: Path,
    output_dir: Path,
    algorithm: str,
) -> CompressionResult:
    logging.debug("Compressing %s using %s", input_path, algorithm)
    text = input_path.read_text(encoding="utf-8")
    original_size = len(text.encode("utf-8"))

    payload_path = output_dir / f"{input_path.stem}.{algorithm}.bin"
    metadata_path = output_dir / f"{input_path.stem}.{algorithm}.json"

    start = time.perf_counter()
    if algorithm == "huffman":
        payload_bytes, bit_len, codes = ruscomp.huffman_encode(text)
        duration = time.perf_counter() - start

        payload_path.write_bytes(bytes(payload_bytes))

        metadata = {
            "algorithm": "huffman",
            "bit_length": bit_len,
            "codes": codes,
            "input_filename": input_path.name,
            "original_size": original_size,
            "payload_size": len(payload_bytes),
            "payload_path": payload_path.name,
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        compressed_size = len(payload_bytes)
    elif algorithm == "rle":
        encoded_text = ruscomp.rle_encode(text)
        duration = time.perf_counter() - start

        payload_bytes = encoded_text.encode("utf-8")
        payload_path.write_bytes(payload_bytes)

        metadata = {
            "algorithm": "rle",
            "input_filename": input_path.name,
            "original_size": original_size,
            "payload_size": len(payload_bytes),
            "payload_path": payload_path.name,
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        compressed_size = len(payload_bytes)
    else:
        raise SystemExit(f"Unsupported algorithm: {algorithm}")

    logging.info(
        "Compressed %s → %s (%.2f%% ratio, %.2f MB/s)",
        input_path,
        payload_path,
        (compressed_size / original_size * 100) if original_size else 0.0,
        (original_size / (1024 * 1024) / duration) if duration else float("inf"),
    )

    return CompressionResult(
        input_path=input_path,
        payload_path=payload_path,
        metadata_path=metadata_path,
        original_bytes=original_size,
        compressed_bytes=compressed_size,
        duration=duration,
    )


def decompress_file(
    metadata_path: Path,
    output_path: Path,
) -> Tuple[int, float]:
    logging.debug("Decompressing using metadata %s", metadata_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    algorithm = metadata.get("algorithm")
    payload_relative = metadata.get("payload_path")
    if not payload_relative:
        raise SystemExit(f"Metadata file {metadata_path} missing payload_path field.")

    payload_path = metadata_path.parent / payload_relative
    payload = payload_path.read_bytes()

    start = time.perf_counter()
    if algorithm == "huffman":
        bit_len = metadata["bit_length"]
        codes: Dict[str, str] = metadata["codes"]
        decoded_text = ruscomp.huffman_decode(payload, bit_len, codes)
    elif algorithm == "rle":
        decoded_text = ruscomp.rle_decode(payload.decode("utf-8"))
    else:
        raise SystemExit(f"Unsupported algorithm in metadata: {algorithm}")
    duration = time.perf_counter() - start

    output_path.write_text(decoded_text, encoding="utf-8")
    logging.info(
        "Decompressed %s → %s in %.3fs",
        metadata_path,
        output_path,
        duration,
    )

    return len(decoded_text.encode("utf-8")), duration


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ruscomp CLI")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    parser.add_argument("--quiet", action="store_true", help="Suppress info logs.")

    subparsers = parser.add_subparsers(dest="command", required=True)

    compress_parser = subparsers.add_parser("compress", help="Compress one or more files.")
    compress_parser.add_argument("inputs", nargs="+", type=Path, help="Input text files.")
    compress_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
        help="Directory to store compressed artifacts (default: current directory).",
    )
    compress_parser.add_argument(
        "--algorithm",
        choices=["huffman", "rle"],
        default="huffman",
        help="Compression algorithm to use.",
    )

    decompress_parser = subparsers.add_parser("decompress", help="Decompress using metadata.")
    decompress_parser.add_argument(
        "metadata",
        nargs="+",
        type=Path,
        help="Metadata JSON file(s) produced during compression.",
    )
    decompress_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
        help="Directory to write decompressed text files.",
    )

    return parser.parse_args(list(argv))


def cmd_compress(args: argparse.Namespace) -> None:
    ensure_output_directory(args.output_dir)

    results: List[CompressionResult] = []
    for input_path in args.inputs:
        if not input_path.exists():
            logging.error("Input path %s does not exist; skipping.", input_path)
            continue
        if not input_path.is_file():
            logging.warning("Input path %s is not a file; skipping.", input_path)
            continue

        result = compress_file(input_path, args.output_dir, args.algorithm)
        results.append(result)

    if not results:
        logging.warning("No files were processed.")
        return

    total_original = sum(r.original_bytes for r in results)
    total_compressed = sum(r.compressed_bytes for r in results)
    total_duration = sum(r.duration for r in results)

    if logging.getLogger().level <= logging.INFO:
        print("\nSummary:")
        for res in results:
            ratio_pct = res.compression_ratio * 100 if res.original_bytes else 0.0
            print(
                f"- {res.input_path.name}: {ratio_pct:.2f}% "
                f"({res.original_bytes}B → {res.compressed_bytes}B), "
                f"{res.speed_mbps:.2f} MB/s"
            )
        overall_ratio = (total_compressed / total_original * 100) if total_original else 0.0
        overall_speed = (
            (total_original / (1024 * 1024)) / total_duration if total_duration else float("inf")
        )
        print(
            f"Total: {overall_ratio:.2f}% "
            f"({total_original}B → {total_compressed}B) "
            f"in {total_duration:.3f}s ({overall_speed:.2f} MB/s)"
        )


def cmd_decompress(args: argparse.Namespace) -> None:
    ensure_output_directory(args.output_dir)

    for metadata_path in args.metadata:
        if not metadata_path.exists():
            logging.error("Metadata path %s does not exist; skipping.", metadata_path)
            continue
        if not metadata_path.is_file():
            logging.warning("Metadata path %s is not a file; skipping.", metadata_path)
            continue

        suffix = metadata_path.stem
        output_path = args.output_dir / f"{suffix}_decoded.txt"
        original_bytes, duration = decompress_file(metadata_path, output_path)

        if logging.getLogger().level <= logging.INFO:
            print(
                f"- {metadata_path.name}: restored {original_bytes}B "
                f"in {duration:.3f}s → {output_path}"
            )


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv or sys.argv[1:])
    configure_logging(args)

    if args.command == "compress":
        cmd_compress(args)
    elif args.command == "decompress":
        cmd_decompress(args)
    else:
        raise SystemExit(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
