"""Pipeline reproduzivel de validacao cruzada agrupada para o UFRB-ESD.

As metricas cientificas sao calculadas exclusivamente nos dez folds externos.
Uma divisao agrupada interna e usada apenas para early stopping. Os mesmos
folds externos sao compartilhados por todas as arquiteturas, permitindo a
comparacao pareada dos resultados.
"""

from __future__ import annotations

import gc
import hashlib
import itertools
import json
import os
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from ufrb_esd_grouping import UnionFind


@dataclass
class ExperimentConfig:
    data_dir: Path
    output_dir: Path
    manifest_path: Path
    feature_cache_dir: Path | None = None
    seed: int = 42
    n_splits: int = 10
    inner_splits: int = 9
    sample_rate: int = 16_000
    duration_seconds: float = 1.0
    segment_hop_seconds: float = 0.1
    n_mfcc: int = 13
    n_mels: int = 40
    frame_length: int = 1_024
    frame_step: int = 512
    batch_size: int = 64
    learning_rate: float = 1e-3
    max_epochs: int = 100
    patience: int = 20
    models: tuple[str, ...] = ("MLP", "CNN2D", "ResNet18", "ResNet50")
    keep_fold_models: bool = False
    resume: bool = True
    quick_test: bool = False
    quick_test_epochs: int = 2
    verbose: int = 2

    def __post_init__(self) -> None:
        self.data_dir = Path(self.data_dir)
        self.output_dir = Path(self.output_dir)
        self.manifest_path = Path(self.manifest_path)
        self.feature_cache_dir = (
            Path(self.feature_cache_dir)
            if self.feature_cache_dir is not None
            else self.output_dir / "feature_cache"
        )
        if self.n_splits != 10 and not self.quick_test:
            raise ValueError("O experimento final deve usar exatamente 10 folds")
        if self.inner_splits < 2:
            raise ValueError("inner_splits deve ser >= 2")

    @property
    def target_length(self) -> int:
        return int(round(self.sample_rate * self.duration_seconds))

    @property
    def segment_hop(self) -> int:
        return max(1, int(round(self.sample_rate * self.segment_hop_seconds)))

    @property
    def effective_epochs(self) -> int:
        return self.quick_test_epochs if self.quick_test else self.max_epochs

    def serializable(self) -> dict[str, Any]:
        values = asdict(self)
        for key in ("data_dir", "output_dir", "manifest_path", "feature_cache_dir"):
            values[key] = str(values[key])
        values["models"] = list(self.models)
        return values


def set_global_determinism(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
    random.seed(seed)
    np.random.seed(seed)
    import tensorflow as tf

    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        # Algumas combinacoes antigas de TensorFlow/GPU nao expoem esta API.
        pass


def load_manifest(path: str | Path) -> pd.DataFrame:
    manifest = pd.read_csv(path, dtype={"content_sha256": str})
    required = {
        "relative_path",
        "class_name",
        "group_id",
        "content_sha256",
    }
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"Manifesto sem colunas obrigatorias: {sorted(missing)}")
    if manifest["relative_path"].duplicated().any():
        raise ValueError("Manifesto possui caminhos duplicados")
    return manifest.sort_values("relative_path", kind="stable").reset_index(drop=True)


def _dataset_fingerprint(manifest: pd.DataFrame, config: ExperimentConfig) -> str:
    digest = hashlib.sha256()
    feature_config = {
        "feature_pipeline_version": 2,
        "sample_rate": config.sample_rate,
        "duration_seconds": config.duration_seconds,
        "segment_hop_seconds": config.segment_hop_seconds,
        "n_mfcc": config.n_mfcc,
        "n_mels": config.n_mels,
        "frame_length": config.frame_length,
        "frame_step": config.frame_step,
    }
    digest.update(json.dumps(feature_config, sort_keys=True).encode("utf-8"))
    for row in manifest.itertuples(index=False):
        digest.update(str(row.relative_path).encode("utf-8"))
        digest.update(str(row.content_sha256).encode("ascii"))
        digest.update(str(row.group_id).encode("utf-8"))
    return digest.hexdigest()


def _loudest_segment(audio: np.ndarray, target_length: int, hop: int) -> np.ndarray:
    if audio.size < target_length:
        return np.pad(audio, (0, target_length - audio.size))
    if audio.size == target_length:
        return audio

    last_start = audio.size - target_length
    starts = list(range(0, last_start + 1, hop))
    if starts[-1] != last_start:
        starts.append(last_start)
    energies = np.asarray(
        [np.sum(np.square(audio[start : start + target_length], dtype=np.float64)) for start in starts]
    )
    best_start = starts[int(np.argmax(energies))]
    return audio[best_start : best_start + target_length]


def _extract_one_feature(path: Path, config: ExperimentConfig) -> tuple[np.ndarray, dict[str, float]]:
    import librosa
    import soundfile as sf

    source_info = sf.info(path)
    source_audio, source_sample_rate = sf.read(
        path, dtype="float64", always_2d=True
    )
    if source_audio.size == 0:
        raise ValueError(f"Audio vazio: {path}")
    # Os trilhos de PCM inteiro nao sao simetricos. Por exemplo, em 16 bits o
    # limite positivo e +32767, enquanto o negativo e -32768. O teste separado
    # evita contar -32767 como clipping e tambem respeita arquivos de 24/32 bits.
    # A medicao acontece antes da reamostragem para nao confundir overshoot do
    # filtro com clipping que ja estava no arquivo.
    subtype_bits = {"PCM_S8": 8, "PCM_16": 16, "PCM_24": 24, "PCM_32": 32}
    pcm_bits = subtype_bits.get(source_info.subtype)
    if pcm_bits is None:
        positive_rail = 1.0
    else:
        positive_rail = 1.0 - 1.0 / float(2 ** (pcm_bits - 1))
    negative_rail = -1.0
    raw_peak = float(np.max(np.abs(source_audio)))
    clip_fraction = float(
        np.mean(
            (source_audio >= positive_rail)
            | (source_audio <= negative_rail)
        )
    )
    near_rail_fraction = float(np.mean(np.abs(source_audio) >= 0.999))
    audio, _ = librosa.load(path, sr=config.sample_rate, mono=True, dtype=np.float32)
    if audio.size == 0:
        raise ValueError(f"Audio vazio: {path}")
    resampled_peak = float(np.max(np.abs(audio)))
    processed = _loudest_segment(audio, config.target_length, config.segment_hop)

    # Forma correta: n_mfcc seleciona coeficientes; o outro eixo e o tempo.
    mfcc = librosa.feature.mfcc(
        y=processed,
        sr=config.sample_rate,
        n_mfcc=config.n_mfcc,
        n_mels=config.n_mels,
        n_fft=config.frame_length,
        win_length=config.frame_length,
        hop_length=config.frame_step,
        center=False,
    ).astype(np.float32)
    feature = mfcc[..., np.newaxis]
    metadata = {
        "loaded_samples": int(audio.size),
        "processed_samples": int(processed.size),
        "source_samples_per_channel": int(source_audio.shape[0]),
        "source_channels": int(source_audio.shape[1]),
        "source_sample_rate": int(source_sample_rate),
        "raw_peak": raw_peak,
        "clip_fraction": clip_fraction,
        "near_rail_fraction": near_rail_fraction,
        "resampled_peak": resampled_peak,
        "mfcc_frames": int(mfcc.shape[1]),
    }
    return feature, metadata


def _merge_feature_duplicate_groups(
    manifest: pd.DataFrame, features: np.ndarray
) -> tuple[pd.DataFrame, dict[str, int]]:
    manifest = manifest.copy()
    feature_hashes = [
        hashlib.sha256(np.ascontiguousarray(feature).tobytes()).hexdigest() for feature in features
    ]
    manifest["feature_sha256"] = feature_hashes
    manifest["group_id_before_feature_merge"] = manifest["group_id"].astype(str)

    union_find = UnionFind(manifest["group_id"].astype(str))
    for _, duplicate_rows in manifest.groupby("feature_sha256", sort=False):
        labels = set(duplicate_rows["class_name"].astype(str))
        if len(labels) > 1:
            paths = duplicate_rows["relative_path"].tolist()
            raise ValueError(
                "Features identicas possuem rotulos conflitantes. Revise os arquivos: " + str(paths)
            )
        group_ids = sorted(set(duplicate_rows["group_id"].astype(str)))
        for group_id in group_ids[1:]:
            union_find.union(group_ids[0], group_id)

    manifest["group_id"] = [union_find.find(str(value)) for value in manifest["group_id"]]
    counts = manifest["feature_sha256"].value_counts()
    manifest["is_feature_duplicate"] = manifest["feature_sha256"].map(counts).gt(1).astype(int)
    duplicate_counts = counts[counts > 1]
    audit = {
        "n_feature_duplicate_groups": int(duplicate_counts.size),
        "n_feature_duplicate_rows": int(duplicate_counts.sum()),
        "n_groups_after_feature_merge": int(manifest["group_id"].nunique()),
    }
    return manifest, audit


def load_or_extract_features(
    config: ExperimentConfig, force: bool = False
) -> tuple[np.ndarray, pd.DataFrame, dict[str, Any]]:
    cache_dir = Path(config.feature_cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "features_corrected.npz"
    cache_manifest_path = cache_dir / "manifest_with_features.csv"
    cache_metadata_path = cache_dir / "features_metadata.json"

    source_manifest = load_manifest(config.manifest_path)
    fingerprint = _dataset_fingerprint(source_manifest, config)
    if not force and cache_path.exists() and cache_manifest_path.exists() and cache_metadata_path.exists():
        metadata = json.loads(cache_metadata_path.read_text(encoding="utf-8"))
        if metadata.get("dataset_fingerprint") == fingerprint:
            features = np.load(cache_path, allow_pickle=False)["features"]
            manifest = pd.read_csv(cache_manifest_path, dtype={"content_sha256": str})
            if len(features) != len(manifest):
                raise ValueError("Cache inconsistente: features e manifesto tem tamanhos diferentes")
            return features, manifest, metadata

    features: list[np.ndarray] = []
    extraction_rows: list[dict[str, float]] = []
    expected_shape: tuple[int, ...] | None = None
    for position, row in enumerate(source_manifest.itertuples(index=False), start=1):
        path = config.data_dir / row.relative_path
        feature, extraction_metadata = _extract_one_feature(path, config)
        if expected_shape is None:
            expected_shape = feature.shape
        elif feature.shape != expected_shape:
            raise ValueError(
                f"Shape MFCC inconsistente para {path}: {feature.shape} != {expected_shape}"
            )
        features.append(feature)
        extraction_rows.append(extraction_metadata)
        if position % 100 == 0 or position == len(source_manifest):
            print(f"MFCC: {position}/{len(source_manifest)}")

    feature_array = np.stack(features).astype(np.float32)
    manifest = pd.concat(
        [source_manifest, pd.DataFrame(extraction_rows)], axis=1, verify_integrity=True
    )
    manifest, duplicate_audit = _merge_feature_duplicate_groups(manifest, feature_array)
    metadata: dict[str, Any] = {
        "dataset_fingerprint": fingerprint,
        "feature_shape": list(feature_array.shape),
        "config": config.serializable(),
        **duplicate_audit,
    }
    np.savez_compressed(cache_path, features=feature_array)
    manifest.to_csv(cache_manifest_path, index=False)
    cache_metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return feature_array, manifest, metadata


def encode_labels(manifest: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    classes = sorted(manifest["class_name"].astype(str).unique())
    class_to_index = {class_name: index for index, class_name in enumerate(classes)}
    labels = manifest["class_name"].map(class_to_index).to_numpy(dtype=np.int64)
    return labels, classes


def create_grouped_folds(
    manifest: pd.DataFrame,
    labels: np.ndarray,
    config: ExperimentConfig,
    force: bool = False,
) -> np.ndarray:
    from sklearn.model_selection import StratifiedGroupKFold

    config.output_dir.mkdir(parents=True, exist_ok=True)
    assignments_path = config.output_dir / "fold_assignments.csv"
    groups = manifest["group_id"].astype(str).to_numpy()
    if assignments_path.exists() and not force:
        saved = pd.read_csv(assignments_path)
        if saved["relative_path"].tolist() != manifest["relative_path"].tolist():
            raise ValueError("fold_assignments.csv nao corresponde ao manifesto atual")
        if saved["group_id"].astype(str).tolist() != manifest["group_id"].astype(str).tolist():
            raise ValueError("Os grupos mudaram; recrie os folds com force=True")
        if "fold_seed" in saved and not saved["fold_seed"].eq(config.seed).all():
            raise ValueError("A semente mudou; recrie os folds com force=True")
        folds = saved["outer_fold"].to_numpy(dtype=int)
    else:
        splitter = StratifiedGroupKFold(
            n_splits=config.n_splits, shuffle=True, random_state=config.seed
        )
        folds = np.full(len(manifest), -1, dtype=int)
        for fold, (_, test_indices) in enumerate(splitter.split(np.zeros(len(labels)), labels, groups)):
            folds[test_indices] = fold
        if np.any(folds < 0):
            raise RuntimeError("Nem todas as amostras receberam um fold")
        output = manifest[["relative_path", "class_name", "group_id"]].copy()
        output["outer_fold"] = folds
        output["fold_seed"] = config.seed
        output.to_csv(assignments_path, index=False)

    _assert_fold_integrity(manifest, labels, folds, config.n_splits)
    return folds


def _assert_fold_integrity(
    manifest: pd.DataFrame, labels: np.ndarray, folds: np.ndarray, n_splits: int
) -> None:
    if set(np.unique(folds)) != set(range(n_splits)):
        raise ValueError(f"Folds encontrados: {sorted(np.unique(folds))}")
    group_fold_counts = (
        pd.DataFrame({"group": manifest["group_id"].astype(str), "fold": folds})
        .groupby("group")["fold"]
        .nunique()
    )
    if (group_fold_counts != 1).any():
        raise AssertionError("Vazamento: um grupo aparece em mais de um fold externo")
    expected_classes = set(np.unique(labels))
    for fold in range(n_splits):
        fold_classes = set(np.unique(labels[folds == fold]))
        if fold_classes != expected_classes:
            raise ValueError(
                f"Fold {fold + 1} nao contem todas as classes: {sorted(fold_classes)}"
            )


def fold_distribution(manifest: pd.DataFrame, folds: np.ndarray) -> pd.DataFrame:
    frame = manifest[["class_name", "group_id"]].copy()
    frame["outer_fold"] = folds + 1
    sample_counts = frame.pivot_table(
        index="outer_fold", columns="class_name", values="group_id", aggfunc="size", fill_value=0
    )
    sample_counts["total_samples"] = sample_counts.sum(axis=1)
    sample_counts["total_groups"] = frame.groupby("outer_fold")["group_id"].nunique()
    return sample_counts


def _inner_grouped_split(
    candidate_indices: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    config: ExperimentConfig,
    fold: int,
) -> tuple[np.ndarray, np.ndarray]:
    from sklearn.model_selection import StratifiedGroupKFold

    inner_splitter = StratifiedGroupKFold(
        n_splits=config.inner_splits,
        shuffle=True,
        random_state=config.seed + fold + 1,
    )
    relative_train, relative_val = next(
        inner_splitter.split(
            np.zeros(len(candidate_indices)),
            labels[candidate_indices],
            groups[candidate_indices],
        )
    )
    train_indices = candidate_indices[relative_train]
    val_indices = candidate_indices[relative_val]
    if set(groups[train_indices]) & set(groups[val_indices]):
        raise AssertionError("Vazamento entre treino interno e validacao interna")
    if set(np.unique(labels[val_indices])) != set(np.unique(labels)):
        raise ValueError(f"Validacao interna do fold {fold + 1} nao contem todas as classes")
    return train_indices, val_indices


def _normalization_statistics(
    features: np.ndarray, train_indices: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    # Media/desvio por coeficiente MFCC; nenhum dado de validacao/teste participa.
    mean = features[train_indices].mean(axis=(0, 2, 3), keepdims=True)
    std = features[train_indices].std(axis=(0, 2, 3), keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def _make_dataset(
    features: np.ndarray,
    labels: np.ndarray,
    indices: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    config: ExperimentConfig,
    training: bool,
    seed: int,
):
    import tensorflow as tf

    x = (features[indices] - mean) / std
    y = labels[indices]
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    if training:
        dataset = dataset.shuffle(len(indices), seed=seed, reshuffle_each_iteration=True)
    dataset = dataset.batch(config.batch_size)
    return dataset.prefetch(tf.data.AUTOTUNE)


def _residual_block(x, filters: int, strides: tuple[int, int] = (1, 1)):
    from tensorflow.keras import layers

    shortcut = x
    y = layers.Conv2D(filters, 3, strides=strides, padding="same", use_bias=False)(x)
    y = layers.BatchNormalization()(y)
    y = layers.Activation("relu")(y)
    y = layers.Conv2D(filters, 3, padding="same", use_bias=False)(y)
    y = layers.BatchNormalization()(y)
    if strides != (1, 1) or int(shortcut.shape[-1]) != filters:
        shortcut = layers.Conv2D(filters, 1, strides=strides, padding="same", use_bias=False)(
            shortcut
        )
        shortcut = layers.BatchNormalization()(shortcut)
    return layers.Activation("relu")(layers.Add()([y, shortcut]))


def _bottleneck_block(x, filters: int, strides: tuple[int, int] = (1, 1)):
    from tensorflow.keras import layers

    shortcut = x
    y = layers.Conv2D(filters, 1, use_bias=False)(x)
    y = layers.BatchNormalization()(y)
    y = layers.Activation("relu")(y)
    y = layers.Conv2D(filters, 3, strides=strides, padding="same", use_bias=False)(y)
    y = layers.BatchNormalization()(y)
    y = layers.Activation("relu")(y)
    y = layers.Conv2D(filters * 4, 1, use_bias=False)(y)
    y = layers.BatchNormalization()(y)
    if strides != (1, 1) or int(shortcut.shape[-1]) != filters * 4:
        shortcut = layers.Conv2D(
            filters * 4, 1, strides=strides, padding="same", use_bias=False
        )(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)
    return layers.Activation("relu")(layers.Add()([y, shortcut]))


def create_model(
    model_name: str, input_shape: tuple[int, ...], num_classes: int, learning_rate: float
):
    import tensorflow as tf
    from tensorflow.keras import layers, models

    if model_name == "MLP":
        model = models.Sequential(
            [
                layers.Input(shape=input_shape),
                layers.Flatten(),
                layers.Dense(128, activation="relu"),
                layers.BatchNormalization(),
                layers.Dropout(0.5),
                layers.Dense(64, activation="relu"),
                layers.BatchNormalization(),
                layers.Dropout(0.5),
                layers.Dense(num_classes, activation="softmax"),
            ],
            name="MLP",
        )
    elif model_name == "CNN2D":
        model = models.Sequential(
            [
                layers.Input(shape=input_shape),
                layers.Conv2D(32, 2, activation="relu"),
                layers.BatchNormalization(),
                layers.MaxPooling2D(2),
                layers.Dropout(0.25),
                layers.Conv2D(64, 2, activation="relu"),
                layers.BatchNormalization(),
                layers.MaxPooling2D(2),
                layers.Dropout(0.25),
                layers.Conv2D(128, 2, activation="relu"),
                layers.BatchNormalization(),
                layers.GlobalMaxPooling2D(),
                layers.Dropout(0.25),
                layers.Dense(256, activation="relu"),
                layers.BatchNormalization(),
                layers.Dropout(0.5),
                layers.Dense(num_classes, activation="softmax"),
            ],
            name="CNN2D",
        )
    elif model_name in {"ResNet18", "ResNet50"}:
        inputs = layers.Input(shape=input_shape)
        x = layers.Conv2D(64, 7, strides=2, padding="same", use_bias=False)(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)
        if model_name == "ResNet18":
            for stage, (filters, repetitions) in enumerate(((64, 2), (128, 2), (256, 2), (512, 2))):
                for repetition in range(repetitions):
                    strides = (2, 2) if stage > 0 and repetition == 0 else (1, 1)
                    x = _residual_block(x, filters, strides)
        else:
            for stage, (filters, repetitions) in enumerate(((64, 3), (128, 4), (256, 6), (512, 3))):
                for repetition in range(repetitions):
                    strides = (2, 2) if stage > 0 and repetition == 0 else (1, 1)
                    x = _bottleneck_block(x, filters, strides)
        x = layers.GlobalAveragePooling2D()(x)
        outputs = layers.Dense(num_classes, activation="softmax")(x)
        model = models.Model(inputs, outputs, name=model_name)
    else:
        raise ValueError(f"Modelo desconhecido: {model_name}")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def _classification_metrics(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import (
        accuracy_score,
        log_loss,
        precision_recall_fscore_support,
        roc_auc_score,
    )

    y_pred = probabilities.argmax(axis=1)
    weighted = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    macro = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    return {
        "loss": float(log_loss(y_true, probabilities, labels=np.arange(probabilities.shape[1]))),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_weighted": float(weighted[0]),
        "recall_weighted": float(weighted[1]),
        "f1_weighted": float(weighted[2]),
        "precision_macro": float(macro[0]),
        "recall_macro": float(macro[1]),
        "f1_macro": float(macro[2]),
        "roc_auc_macro_ovr": float(
            roc_auc_score(y_true, probabilities, multi_class="ovr", average="macro")
        ),
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def run_grouped_cross_validation(
    features: np.ndarray,
    manifest: pd.DataFrame,
    config: ExperimentConfig,
    force_folds: bool = False,
) -> pd.DataFrame:
    import tensorflow as tf

    set_global_determinism(config.seed)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config_path = config.output_dir / "experiment_config.json"
    current_config = config.serializable()
    if config_path.exists() and (config.output_dir / "runs").exists():
        previous_config = json.loads(config_path.read_text(encoding="utf-8"))
        if previous_config != current_config:
            raise ValueError(
                "A configuracao mudou, mas existem resultados retomaveis neste diretorio. "
                "Use outro output_dir ou remova conscientemente a execucao anterior."
            )
    _write_json(config_path, current_config)

    labels, class_names = encode_labels(manifest)
    groups = manifest["group_id"].astype(str).to_numpy()
    folds = create_grouped_folds(manifest, labels, config, force=force_folds)
    fold_distribution(manifest, folds).to_csv(config.output_dir / "fold_distribution.csv")
    (config.output_dir / "class_names.json").write_text(
        json.dumps(class_names, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    selected_models = ("MLP",) if config.quick_test else config.models
    all_indices = np.arange(len(manifest))
    metric_records: list[dict[str, Any]] = []

    for fold in range(config.n_splits):
        test_indices = all_indices[folds == fold]
        outer_train_indices = all_indices[folds != fold]
        train_indices, val_indices = _inner_grouped_split(
            outer_train_indices, labels, groups, config, fold
        )
        if set(groups[outer_train_indices]) & set(groups[test_indices]):
            raise AssertionError("Vazamento entre treino e teste externos")
        mean, std = _normalization_statistics(features, train_indices)

        for model_position, model_name in enumerate(selected_models):
            run_dir = config.output_dir / "runs" / f"fold_{fold + 1:02d}" / model_name
            run_dir.mkdir(parents=True, exist_ok=True)
            metrics_path = run_dir / "metrics.json"
            predictions_path = run_dir / "predictions.csv"
            if config.resume and metrics_path.exists() and predictions_path.exists():
                metric_records.append(json.loads(metrics_path.read_text(encoding="utf-8")))
                print(f"[retomado] fold={fold + 1} modelo={model_name}")
                continue

            run_seed = config.seed + fold * 100 + model_position
            set_global_determinism(run_seed)
            tf.keras.backend.clear_session()
            train_ds = _make_dataset(
                features, labels, train_indices, mean, std, config, True, run_seed
            )
            val_ds = _make_dataset(
                features, labels, val_indices, mean, std, config, False, run_seed
            )
            test_ds = _make_dataset(
                features, labels, test_indices, mean, std, config, False, run_seed
            )
            model = create_model(
                model_name,
                input_shape=tuple(features.shape[1:]),
                num_classes=len(class_names),
                learning_rate=config.learning_rate,
            )
            checkpoint_path = run_dir / "best.weights.h5"
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=config.patience,
                    restore_best_weights=False,
                ),
                tf.keras.callbacks.ModelCheckpoint(
                    checkpoint_path,
                    monitor="val_loss",
                    save_best_only=True,
                    save_weights_only=True,
                ),
                tf.keras.callbacks.TerminateOnNaN(),
            ]
            print(f"\n=== Fold {fold + 1}/{config.n_splits} | {model_name} ===")
            history = model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=config.effective_epochs,
                callbacks=callbacks,
                verbose=config.verbose,
            )
            model.load_weights(checkpoint_path)
            probabilities = model.predict(test_ds, verbose=0)
            fold_metrics = _classification_metrics(labels[test_indices], probabilities)
            best_epoch = int(np.argmin(history.history["val_loss"]) + 1)
            record: dict[str, Any] = {
                "fold": fold + 1,
                "model": model_name,
                "seed": run_seed,
                "n_train": int(len(train_indices)),
                "n_validation": int(len(val_indices)),
                "n_test": int(len(test_indices)),
                "n_test_groups": int(len(set(groups[test_indices]))),
                "epochs_executed": int(len(history.history["loss"])),
                "best_epoch": best_epoch,
                "parameters": int(model.count_params()),
                **fold_metrics,
            }
            _write_json(metrics_path, record)
            pd.DataFrame(history.history).assign(epoch=lambda frame: np.arange(1, len(frame) + 1)).to_csv(
                run_dir / "history.csv", index=False
            )
            np.savez(run_dir / "normalization.npz", mean=mean, std=std)

            predictions = manifest.loc[
                test_indices, ["relative_path", "class_name", "group_id"]
            ].copy()
            predictions.insert(0, "sample_index", test_indices)
            predictions.insert(1, "fold", fold + 1)
            predictions.insert(2, "model", model_name)
            predictions["true_index"] = labels[test_indices]
            predictions["predicted_index"] = probabilities.argmax(axis=1)
            predictions["predicted_class"] = [
                class_names[index] for index in probabilities.argmax(axis=1)
            ]
            for class_index, class_name in enumerate(class_names):
                predictions[f"prob_{class_name}"] = probabilities[:, class_index]
            predictions.to_csv(predictions_path, index=False)
            metric_records.append(record)

            if not config.keep_fold_models:
                checkpoint_path.unlink(missing_ok=True)
            del model, train_ds, val_ds, test_ds, probabilities
            tf.keras.backend.clear_session()
            gc.collect()

    metrics = pd.DataFrame(metric_records).sort_values(["model", "fold"])
    metrics.to_csv(config.output_dir / "fold_metrics.csv", index=False)
    return metrics


METRIC_COLUMNS = (
    "accuracy",
    "precision_weighted",
    "recall_weighted",
    "f1_weighted",
    "precision_macro",
    "recall_macro",
    "f1_macro",
    "roc_auc_macro_ovr",
)


def _holm_adjust(p_values: Iterable[float]) -> list[float]:
    values = np.asarray(list(p_values), dtype=float)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running_max = 0.0
    total = len(values)
    for rank, index in enumerate(order):
        candidate = min(1.0, (total - rank) * values[index])
        running_max = max(running_max, candidate)
        adjusted[index] = running_max
    return adjusted.tolist()


def summarize_experiment(config: ExperimentConfig) -> dict[str, pd.DataFrame]:
    from scipy import stats
    from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
    import matplotlib.pyplot as plt
    import seaborn as sns

    metrics = pd.read_csv(config.output_dir / "fold_metrics.csv")
    expected_folds = set(range(1, config.n_splits + 1))
    for model_name, model_rows in metrics.groupby("model"):
        if set(model_rows["fold"]) != expected_folds:
            raise ValueError(f"Resultados incompletos para {model_name}")

    summary_records: list[dict[str, Any]] = []
    for model_name, model_rows in metrics.groupby("model", sort=False):
        for metric in METRIC_COLUMNS:
            values = model_rows.sort_values("fold")[metric].to_numpy(float)
            mean = float(np.mean(values))
            std = float(np.std(values, ddof=1))
            margin = float(stats.t.ppf(0.975, len(values) - 1) * std / np.sqrt(len(values)))
            summary_records.append(
                {
                    "model": model_name,
                    "metric": metric,
                    "mean": mean,
                    "std": std,
                    "ci95_low": mean - margin,
                    "ci95_high": mean + margin,
                    "n_folds": len(values),
                }
            )
    summary = pd.DataFrame(summary_records)
    summary.to_csv(config.output_dir / "summary_mean_std_ci.csv", index=False)

    predictions = pd.concat(
        [pd.read_csv(path) for path in sorted((config.output_dir / "runs").glob("fold_*/*/predictions.csv"))],
        ignore_index=True,
    )
    predictions.to_csv(config.output_dir / "oof_predictions.csv", index=False)
    class_names = json.loads((config.output_dir / "class_names.json").read_text(encoding="utf-8"))

    report_rows: list[pd.DataFrame] = []
    for model_name, model_predictions in predictions.groupby("model", sort=False):
        y_true = model_predictions["true_index"].to_numpy(int)
        y_pred = model_predictions["predicted_index"].to_numpy(int)
        report = pd.DataFrame(
            classification_report(
                y_true,
                y_pred,
                labels=np.arange(len(class_names)),
                target_names=class_names,
                output_dict=True,
                zero_division=0,
            )
        ).T.reset_index(names="class_or_average")
        report.insert(0, "model", model_name)
        report_rows.append(report)

        matrix = confusion_matrix(y_true, y_pred, labels=np.arange(len(class_names)))
        plt.figure(figsize=(11, 9))
        sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
        plt.xlabel("Predito")
        plt.ylabel("Real")
        plt.title(f"Matriz de confusao OOF - {model_name}")
        plt.tight_layout()
        plt.savefig(config.output_dir / f"confusion_oof_{model_name}.png", dpi=300)
        plt.close()

    reports = pd.concat(report_rows, ignore_index=True)
    reports.to_csv(config.output_dir / "oof_classification_reports.csv", index=False)

    # Curvas ROC OOF macro: cada amostra foi prevista exatamente uma vez por modelo.
    plt.figure(figsize=(10, 8))
    for model_name, model_predictions in predictions.groupby("model", sort=False):
        y_true = model_predictions["true_index"].to_numpy(int)
        probabilities = model_predictions[[f"prob_{name}" for name in class_names]].to_numpy(float)
        y_binary = np.eye(len(class_names), dtype=int)[y_true]
        fprs: dict[int, np.ndarray] = {}
        tprs: dict[int, np.ndarray] = {}
        for class_index in range(len(class_names)):
            fprs[class_index], tprs[class_index], _ = roc_curve(
                y_binary[:, class_index], probabilities[:, class_index]
            )
        all_fpr = np.unique(np.concatenate(list(fprs.values())))
        mean_tpr = np.mean(
            [np.interp(all_fpr, fprs[index], tprs[index]) for index in range(len(class_names))],
            axis=0,
        )
        macro_auc = auc(all_fpr, mean_tpr)
        plt.plot(all_fpr, mean_tpr, label=f"{model_name} (AUC={macro_auc:.4f})")
    plt.plot([0, 1], [0, 1], "k--", label="Aleatório")
    plt.xlabel("Taxa de falsos positivos")
    plt.ylabel("Taxa de verdadeiros positivos")
    plt.title("Curvas ROC OOF — média macro")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(config.output_dir / "roc_oof_comparative.png", dpi=300)
    plt.close()

    # Friedman (omnibus) seguido de Wilcoxon pareado com correcao de Holm.
    statistical_records: list[dict[str, Any]] = []
    model_names = list(metrics["model"].drop_duplicates())
    for metric in METRIC_COLUMNS:
        pivot = metrics.pivot(index="fold", columns="model", values=metric)[model_names]
        if len(model_names) >= 3:
            statistic, p_value = stats.friedmanchisquare(
                *(pivot[model].to_numpy(float) for model in model_names)
            )
            statistical_records.append(
                {
                    "metric": metric,
                    "test": "Friedman",
                    "model_a": "all",
                    "model_b": "all",
                    "statistic": float(statistic),
                    "p_value": float(p_value),
                    "p_holm": np.nan,
                }
            )
        pair_records: list[dict[str, Any]] = []
        for model_a, model_b in itertools.combinations(model_names, 2):
            try:
                statistic, p_value = stats.wilcoxon(
                    pivot[model_a], pivot[model_b], alternative="two-sided", zero_method="zsplit"
                )
            except ValueError:
                statistic, p_value = 0.0, 1.0
            pair_records.append(
                {
                    "metric": metric,
                    "test": "Wilcoxon paired folds",
                    "model_a": model_a,
                    "model_b": model_b,
                    "statistic": float(statistic),
                    "p_value": float(p_value),
                }
            )
        adjusted = _holm_adjust(record["p_value"] for record in pair_records)
        for record, adjusted_p in zip(pair_records, adjusted):
            record["p_holm"] = adjusted_p
            statistical_records.append(record)
    statistics = pd.DataFrame(statistical_records)
    statistics.to_csv(config.output_dir / "paired_statistical_tests.csv", index=False)
    return {
        "fold_metrics": metrics,
        "summary": summary,
        "statistics": statistics,
        "classification_reports": reports,
        "oof_predictions": predictions,
    }


def train_final_model(
    features: np.ndarray,
    manifest: pd.DataFrame,
    config: ExperimentConfig,
    model_name: str,
):
    """Treina o artefato de implantacao em todos os dados apos a avaliacao.

    O numero de epocas e a mediana dos melhores epochs nos folds externos. Este
    modelo final nao produz a metrica do artigo; as metricas validas sao OOF.
    """
    import tensorflow as tf

    labels, class_names = encode_labels(manifest)
    metrics = pd.read_csv(config.output_dir / "fold_metrics.csv")
    model_metrics = metrics[metrics["model"] == model_name]
    if len(model_metrics) != config.n_splits:
        raise ValueError(f"Sao necessarios {config.n_splits} folds completos para {model_name}")
    epochs = max(1, int(round(model_metrics["best_epoch"].median())))
    all_indices = np.arange(len(features))
    mean = features.mean(axis=(0, 2, 3), keepdims=True).astype(np.float32)
    std = features.std(axis=(0, 2, 3), keepdims=True).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std)
    set_global_determinism(config.seed)
    dataset = _make_dataset(
        features, labels, all_indices, mean, std, config, True, config.seed
    )
    model = create_model(
        model_name, tuple(features.shape[1:]), len(class_names), config.learning_rate
    )
    model.fit(dataset, epochs=epochs, verbose=config.verbose)
    final_dir = config.output_dir / "final_model"
    final_dir.mkdir(parents=True, exist_ok=True)
    model.save(final_dir / f"{model_name}.keras")
    np.savez(final_dir / "normalization.npz", mean=mean, std=std)
    _write_json(
        final_dir / "metadata.json",
        {
            "model": model_name,
            "epochs": epochs,
            "class_names": class_names,
            "feature_shape": list(features.shape[1:]),
            "note": "Modelo de implantacao; desempenho deve ser citado a partir das previsoes OOF.",
        },
    )
    return model
