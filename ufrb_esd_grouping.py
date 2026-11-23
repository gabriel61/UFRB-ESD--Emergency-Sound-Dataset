"""Agrupamento auditavel dos arquivos do UFRB-ESD por gravacao de origem.

Este modulo usa apenas a biblioteca padrao para que o manifesto possa ser
gerado e inspecionado antes da instalacao de TensorFlow/librosa. O identificador
de grupo e global (nao inclui a classe), pois uma mesma gravacao pode conter
recortes rotulados em classes diferentes.
"""

from __future__ import annotations

import csv
import hashlib
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


WAV_SUFFIX = ".wav"
TERMINAL_CHUNK_RE = re.compile(r"^(?P<source>.+)_(?P<chunk>\d+)$")


class UnionFind:
    def __init__(self, values: Iterable[str] = ()) -> None:
        self.parent = {value: value for value in values}

    def add(self, value: str) -> None:
        self.parent.setdefault(value, value)

    def find(self, value: str) -> str:
        self.add(value)
        root = value
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[value] != value:
            parent = self.parent[value]
            self.parent[value] = root
            value = parent
        return root

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        # A raiz lexicograficamente menor torna o resultado deterministico.
        root, child = sorted((left_root, right_root))
        self.parent[child] = root


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _numeric_source_map(
    stems: Iterable[str], max_chunk_suffix: int = 19
) -> dict[str, tuple[str, str, bool]]:
    """Infere fontes numericas considerando recortes enumerados de 0 a 19.

    A inferencia precisa observar o conjunto inteiro de nomes. Por exemplo,
    ``226011`` pode ser o recorte 1 de ``22601`` e ``2260110`` o recorte 10;
    analisar cada nome isoladamente confundiria o recorte 1 com uma nova fonte.

    Uma fonte sem arquivo-base e aceita quando ha ao menos dois recortes com
    sufixos de um digito e nenhum desses nomes apresenta descendentes que nao
    possam ser explicados pela mesma sequencia 0..19. Casos sem evidencia
    suficiente permanecem como singletons e sao enviados para revisao.
    """
    if max_chunk_suffix < 1:
        raise ValueError("max_chunk_suffix deve ser >= 1")

    numeric_stems = {stem for stem in stems if stem.isdigit()}
    if not numeric_stems:
        return {}

    suffixes = tuple(str(number) for number in range(0, max_chunk_suffix + 1))
    candidate_bases: set[str] = set(numeric_stems)
    for stem in numeric_stems:
        for suffix in suffixes:
            if stem.endswith(suffix) and len(stem) > len(suffix):
                candidate_bases.add(stem[: -len(suffix)])

    families: dict[str, set[str]] = {}
    accepted_sources: set[str] = set()
    source_kinds: dict[str, str] = {}
    for base in candidate_bases:
        children = {f"{base}{suffix}" for suffix in suffixes} & numeric_stems
        family = children | ({base} if base in numeric_stems else set())
        families[base] = family

        if base in numeric_stems and children:
            accepted_sources.add(base)
            source_kinds[base] = "existing"
            continue

        # Sem o arquivo-base, exigimos uma sequencia inequivoca. Dois sufixos
        # entre 0 e 9 evitam interpretar uma fonte real terminada em "1" e os
        # seus recortes 10..19 como uma fonte ancestral inexistente.
        short_children = {
            f"{base}{number}" for number in range(0, min(9, max_chunk_suffix) + 1)
        } & children
        if base in numeric_stems or len(short_children) < 2:
            continue

        has_external_descendant = False
        for member in children:
            for suffix in suffixes:
                descendant = f"{member}{suffix}"
                if descendant in numeric_stems and descendant not in family:
                    has_external_descendant = True
                    break
            if has_external_descendant:
                break
        if not has_external_descendant:
            accepted_sources.add(base)
            source_kinds[base] = "missing"

    # Um nome explicado como recorte de uma fonte mais curta so vira uma nova
    # fonte quando possui descendentes que a sequencia 0..19 do pai nao explica.
    # Assim, 3034410..3034415 continuam em 30344 (recortes 10..15), enquanto
    # 2434720..2434723 comprovam que 243472 e uma fonte distinta de 24347.
    nested_sources = {
        source
        for source in accepted_sources
        if any(
            len(parent) < len(source) and source in families[parent]
            and not any(child not in families[parent] for child in families[source] - {source})
            for parent in accepted_sources
        )
    }
    accepted_sources.difference_update(nested_sources)
    for source in nested_sources:
        source_kinds.pop(source, None)

    result: dict[str, tuple[str, str, bool]] = {}
    for stem in numeric_stems:
        matches = [base for base in accepted_sources if stem in families[base]]
        if matches:
            # Se o proprio nome e uma fonte aceita, ele prevalece. Para os seus
            # recortes, a fonte aceita mais especifica evita colisao de prefixos.
            source = stem if stem in matches else max(matches, key=lambda value: (len(value), value))
            kind = source_kinds[source]
            if stem == source:
                rule = "numeric_base_existing"
            elif kind == "existing":
                rule = "numeric_cut_existing_base_0_19"
            else:
                rule = "numeric_cut_inferred_missing_base_0_19"
            result[stem] = (f"numeric:{source}", rule, kind == "missing")
        else:
            # Um singleton isolado nao pode causar vazamento por si so. Ele so
            result[stem] = (
                f"numeric:{stem}",
                "numeric_singleton",
                False,
            )
    return result


def infer_source_group(stem: str) -> tuple[str, str, bool]:
    """Infere ``(group_id, rule, needs_review)`` a partir do nome sem extensao.

    Regras, da mais explicita para a mais conservadora:
    1. Freesound: ``<id>__<autor>__<nome>``.
    2. Recortes com delimitador: remove apenas o ultimo ``_<numero>``.
    3. Nomes somente numericos: retorna provisoriamente um singleton. A
       inferencia coletiva 0..19 e feita por :func:`build_manifest_rows`.
    """
    if "__" in stem:
        source_id = stem.split("__", 1)[0]
        if source_id.isdigit():
            return f"freesound:{source_id}", "freesound_id", False

    terminal_match = TERMINAL_CHUNK_RE.fullmatch(stem)
    if terminal_match:
        source = terminal_match.group("source")
        return f"named:{source}", "terminal_numeric_chunk", False

    if stem.isdigit():
        return f"numeric:{stem}", "numeric_singleton", True

    return f"singleton:{stem}", "singleton", True


def read_overrides(path: Path | None) -> dict[str, str]:
    if path is None or not path.exists():
        return {}
    overrides: dict[str, str] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"relative_path", "group_id"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"{path} deve conter as colunas {sorted(required)}")
        for row in reader:
            relative_path = row["relative_path"].replace("\\", "/").strip()
            group_id = row["group_id"].strip()
            if not relative_path or not group_id:
                raise ValueError(f"Override incompleto em {path}: {row}")
            if relative_path in overrides:
                raise ValueError(f"Override duplicado para {relative_path}")
            overrides[relative_path] = group_id
    return overrides


def read_exclusions(path: Path | None) -> dict[str, str]:
    """Le exclusoes documentadas como ``relative_path,reason``."""
    if path is None or not path.exists():
        return {}
    exclusions: dict[str, str] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"relative_path", "reason"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"{path} deve conter as colunas {sorted(required)}")
        for row in reader:
            relative_path = row["relative_path"].replace("\\", "/").strip()
            reason = row["reason"].strip()
            if not relative_path or not reason:
                raise ValueError(f"Exclusao incompleta em {path}: {row}")
            if relative_path in exclusions:
                raise ValueError(f"Exclusao duplicada para {relative_path}")
            exclusions[relative_path] = reason
    return exclusions


def build_manifest_rows(
    data_dir: str | Path,
    overrides_path: str | Path | None = None,
    exclusions_path: str | Path | None = None,
    max_chunk_suffix: int = 19,
) -> list[dict[str, object]]:
    data_dir = Path(data_dir).resolve()
    overrides_file = Path(overrides_path) if overrides_path else None
    overrides = read_overrides(overrides_file)
    exclusions_file = Path(exclusions_path) if exclusions_path else None
    exclusions = read_exclusions(exclusions_file)
    rows: list[dict[str, object]] = []

    wav_paths = sorted(
        (path for path in data_dir.glob("*/*") if path.is_file() and path.suffix.lower() == WAV_SUFFIX),
        key=lambda path: path.relative_to(data_dir).as_posix().casefold(),
    )
    if not wav_paths:
        raise FileNotFoundError(f"Nenhum WAV encontrado em {data_dir}")

    numeric_groups = _numeric_source_map(
        (path.stem for path in wav_paths), max_chunk_suffix=max_chunk_suffix
    )

    discovered_relative_paths: set[str] = set()
    for path in wav_paths:
        relative_path = path.relative_to(data_dir).as_posix()
        discovered_relative_paths.add(relative_path)
        if relative_path in exclusions:
            continue
        inferred_group, rule, needs_review = numeric_groups.get(
            path.stem, infer_source_group(path.stem)
        )
        if relative_path in overrides:
            inferred_group = overrides[relative_path]
            rule = "manual_override"
            needs_review = False
        rows.append(
            {
                "relative_path": relative_path,
                "class_name": path.parent.name,
                "file_name": path.name,
                "stem": path.stem,
                "group_id_inferred": inferred_group,
                "group_rule": rule,
                "needs_review": int(needs_review),
                "file_size": path.stat().st_size,
                "content_sha256": sha256_file(path),
            }
        )

    unknown_overrides = set(overrides) - discovered_relative_paths
    if unknown_overrides:
        raise ValueError(f"Overrides apontam para arquivos inexistentes: {sorted(unknown_overrides)}")
    unknown_exclusions = set(exclusions) - discovered_relative_paths
    if unknown_exclusions:
        raise ValueError(
            f"Exclusoes apontam para arquivos inexistentes: {sorted(unknown_exclusions)}"
        )

    # Une grupos inferidos que contenham arquivos WAV byte a byte identicos.
    union_find = UnionFind(str(row["group_id_inferred"]) for row in rows)
    groups_by_hash: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        groups_by_hash[str(row["content_sha256"])].add(str(row["group_id_inferred"]))
    for group_ids in groups_by_hash.values():
        group_ids = sorted(group_ids)
        for group_id in group_ids[1:]:
            union_find.union(group_ids[0], group_id)

    hash_counts = Counter(str(row["content_sha256"]) for row in rows)
    for row in rows:
        row["group_id"] = union_find.find(str(row["group_id_inferred"]))
        row["is_exact_duplicate"] = int(hash_counts[str(row["content_sha256"])] > 1)
    return rows


def write_manifest_csv(rows: list[dict[str, object]], output_path: str | Path) -> Path:
    if not rows:
        raise ValueError("O manifesto nao pode ser vazio")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def manifest_audit(rows: list[dict[str, object]]) -> dict[str, object]:
    paths = [str(row["relative_path"]) for row in rows]
    if len(paths) != len(set(paths)):
        raise ValueError("Ha caminhos repetidos no manifesto")

    group_to_rows: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        group_to_rows[str(row["group_id"])].append(row)

    class_counts = Counter(str(row["class_name"]) for row in rows)
    groups_per_class: dict[str, int] = {}
    for class_name in sorted(class_counts):
        groups_per_class[class_name] = len(
            {str(row["group_id"]) for row in rows if row["class_name"] == class_name}
        )

    duplicate_hashes = Counter(str(row["content_sha256"]) for row in rows)
    duplicate_groups = [count for count in duplicate_hashes.values() if count > 1]
    multi_class_groups = {
        group_id: sorted({str(row["class_name"]) for row in group_rows})
        for group_id, group_rows in group_to_rows.items()
        if len({str(row["class_name"]) for row in group_rows}) > 1
    }
    return {
        "n_files": len(rows),
        "n_groups": len(group_to_rows),
        "n_singleton_groups": sum(len(group_rows) == 1 for group_rows in group_to_rows.values()),
        "n_multi_file_groups": sum(len(group_rows) > 1 for group_rows in group_to_rows.values()),
        "largest_group_size": max(map(len, group_to_rows.values())),
        "n_rows_marked_for_review": sum(int(row["needs_review"]) for row in rows),
        "n_duplicate_hash_groups": len(duplicate_groups),
        "n_duplicate_files": sum(duplicate_groups),
        "class_counts": dict(sorted(class_counts.items())),
        "groups_per_class": groups_per_class,
        "multi_class_groups": multi_class_groups,
    }
