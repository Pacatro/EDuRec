"""Data build configuration and cache manifest."""

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from edurec import settings
from edurec.datasets.sources.registry import DatasetName

#: Schema version of the persisted processed data. Bump when the layout or the
#: meaning of any stored artefact changes.
PROCESSING_VERSION = "v8"

type SplitStrategy = Literal["temporal", "random"]


def default_min_interactions(dataset: DatasetName) -> int:
    """Minimum support used when the caller does not provide one explicitly."""
    if dataset is DatasetName.MOOCCUBEX:
        return settings.MOOCCUBEX_MIN_CONNECTIONS
    return settings.MIN_INTERACTIONS


def source_filter_options(dataset: DatasetName) -> dict[str, Any]:
    """Source-definition knobs that must invalidate a processed cache."""
    if dataset is not DatasetName.MOOCCUBEX:
        return {}
    return {
        "mooccubex_max_user_enrollments": settings.MOOCCUBEX_MAX_USER_ENROLLMENTS,
        "mooccubex_exclude_deprecated_courses": (
            settings.MOOCCUBEX_EXCLUDE_DEPRECATED_COURSES
        ),
        "mooccubex_experimental_course_pattern": (
            settings.MOOCCUBEX_EXPERIMENTAL_COURSE_PATTERN
        ),
        "mooccubex_educational_resource_prefixes": list(
            settings.MOOCCUBEX_EDUCATIONAL_RESOURCE_PREFIXES
        ),
    }


@dataclass(frozen=True)
class DataConfig:
    """Every option that affects the processed data and its cache."""

    dataset: DatasetName
    test_ratio: float = settings.TEST_RATIO
    val_ratio: float = settings.VAL_RATIO
    min_interactions: int | None = None
    max_interactions: int | None = None
    remove_sparse: bool = settings.REMOVE_SPARSE
    deduplicate: bool = settings.DEDUPLICATE_INTERACTIONS
    random_state: int | None = None
    sampling_seed: int = 42
    max_exclusive_concepts: int | None = None

    def __post_init__(self) -> None:
        if isinstance(self.sampling_seed, bool) or not isinstance(
            self.sampling_seed, int
        ):
            raise TypeError("sampling_seed must be an integer.")
        if self.max_exclusive_concepts is not None:
            if (
                isinstance(self.max_exclusive_concepts, bool)
                or not isinstance(self.max_exclusive_concepts, int)
                or self.max_exclusive_concepts < 0
            ):
                raise ValueError("max_exclusive_concepts must be nonnegative or None.")
            if self.dataset is not DatasetName.MOOCCUBEX:
                raise ValueError(
                    "max_exclusive_concepts is only supported by MOOCCubeX."
                )
        if not 0 < self.test_ratio < 1 or not 0 < self.val_ratio < 1:
            raise ValueError("Validation/test ratios must be in (0, 1).")
        if self.test_ratio + self.val_ratio >= 1:
            raise ValueError("Validation/test ratios must sum to less than 1.")
        min_interactions = (
            default_min_interactions(self.dataset)
            if self.min_interactions is None
            else self.min_interactions
        )
        object.__setattr__(self, "min_interactions", min_interactions)
        if min_interactions < 3:
            raise ValueError(
                "min_interactions must be at least 3 to create train/val/test splits."
            )
        if self.max_interactions is not None:
            if (
                isinstance(self.max_interactions, bool)
                or not isinstance(self.max_interactions, int)
                or self.max_interactions < 1
            ):
                raise ValueError("max_interactions must be a positive integer or None.")
            if self.max_interactions < min_interactions:
                raise ValueError("max_interactions must be at least min_interactions.")

    @property
    def effective_min_interactions(self) -> int:
        """Resolved minimum support after dataset defaults are applied."""
        if self.min_interactions is None:
            raise RuntimeError("min_interactions was not resolved.")
        return self.min_interactions

    def to_dict(self) -> dict[str, Any]:
        """Plain, logger-friendly representation of the configuration."""
        payload = asdict(self)
        payload["dataset"] = self.dataset.value
        return payload

    def manifest(
        self,
        strategy: SplitStrategy,
        num_users: int = 0,
        num_items: int = 0,
    ) -> "BuildManifest":
        options = asdict(self)
        options.pop("dataset", None)
        options.pop("random_state", None)
        options.update(source_filter_options(self.dataset))

        return BuildManifest(
            dataset=self.dataset.value,
            config=options,
            split_seed=self.random_state,
            split_strategy=strategy,
            num_users=num_users,
            num_items=num_items,
        )


@dataclass(frozen=True)
class BuildManifest:
    """Identity of the configuration that produced a processed cache."""

    dataset: str
    config: dict[str, Any] = field(default_factory=dict)
    split_seed: int | None = None
    split_strategy: SplitStrategy = "random"
    num_users: int = 0
    num_items: int = 0
    processing_version: str = PROCESSING_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "BuildManifest":
        split_seed = payload.get("split_seed")

        return cls(
            dataset=str(payload.get("dataset", "")),
            config=dict(payload.get("config", {})),
            split_seed=None if split_seed is None else int(split_seed),
            split_strategy=str(payload.get("split_strategy", "random")),  # type: ignore[arg-type]
            num_users=int(payload.get("num_users", 0) or 0),
            num_items=int(payload.get("num_items", 0) or 0),
            processing_version=str(payload.get("processing_version", "")),
        )

    def incompatibility(self, expected: "BuildManifest") -> str | None:
        """Human-readable reason this cache cannot satisfy ``expected``."""
        if self.dataset != expected.dataset:
            return (
                f"cache belongs to dataset {self.dataset!r}, "
                f"expected {expected.dataset!r}"
            )
        if self.processing_version != expected.processing_version:
            return (
                f"cache uses processing version {self.processing_version!r}, "
                f"expected {expected.processing_version!r}"
            )
        if self.split_strategy != expected.split_strategy:
            return (
                f"cache was split with strategy {self.split_strategy!r}, "
                f"expected {expected.split_strategy!r}"
            )
        if self.config != expected.config:
            return "cache was built with different data options"
        if self.split_strategy == "random" and self.split_seed != expected.split_seed:
            return (
                f"cache was split with seed {self.split_seed!r}, "
                f"expected {expected.split_seed!r}"
            )

        return None
