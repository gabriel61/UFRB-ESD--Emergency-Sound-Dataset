"""Audita saturacao diretamente nas amostras PCM dos WAVs originais."""

from __future__ import annotations

import argparse
import array
import csv
import sys
import wave
from pathlib import Path


def _read_pcm(path: Path) -> tuple[list[int] | array.array, int]:
    with wave.open(str(path), "rb") as stream:
        sample_width = stream.getsampwidth()
        raw = stream.readframes(stream.getnframes())

    if sample_width == 2:
        samples = array.array("h")
        samples.frombytes(raw)
        if sys.byteorder != "little":
            samples.byteswap()
        return samples, 32_767
    if sample_width == 4:
        samples = array.array("i")
        samples.frombytes(raw)
        if sys.byteorder != "little":
            samples.byteswap()
        return samples, 2_147_483_647
    if sample_width == 3:
        samples_24 = []
        for position in range(0, len(raw) - 2, 3):
            value = raw[position] | raw[position + 1] << 8 | raw[position + 2] << 16
            if value & 0x800000:
                value -= 1 << 24
            samples_24.append(value)
        return samples_24, 8_388_607
    raise ValueError(f"largura PCM nao suportada: {sample_width} bytes")


def audit_file(path: Path) -> dict[str, object]:
    samples, positive_limit = _read_pcm(path)
    negative_limit = -positive_limit - 1
    total = len(samples)
    if not total:
        raise ValueError("arquivo sem amostras")
    exact_rail = sum(
        value >= positive_limit or value <= negative_limit for value in samples
    )
    near_rail = sum(abs(value) >= 0.999 * positive_limit for value in samples)
    peak = max(abs(value) for value in samples) / positive_limit
    return {
        "source_pcm_peak": peak,
        "source_exact_rail_fraction": exact_rail / total,
        "source_near_rail_fraction": near_rail / total,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("output_csv", type=Path)
    parser.add_argument("--paths-csv", type=Path)
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    errors: list[tuple[str, str]] = []
    if args.paths_csv:
        with args.paths_csv.open("r", encoding="utf-8-sig", newline="") as stream:
            relative_paths = [row["relative_path"] for row in csv.DictReader(stream)]
        paths = [args.data_dir / relative_path for relative_path in relative_paths]
    else:
        paths = sorted(args.data_dir.glob("*/*.wav"))
    for path in paths:
        relative_path = path.relative_to(args.data_dir).as_posix()
        try:
            result = audit_file(path)
            rows.append({"relative_path": relative_path, **result})
        except Exception as error:  # Mantem formatos nao PCM visiveis no relatorio.
            errors.append((relative_path, str(error)))

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Arquivos auditados: {len(rows)}")
    print(f"Erros: {len(errors)}")
    for relative_path, message in errors:
        print(f"  {relative_path}: {message}")


if __name__ == "__main__":
    main()
