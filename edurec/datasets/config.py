"""Data build configuration and cache manifest."""

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from edurec import settings
from edurec.datasets.registry import DatasetName

#: Schema version of the persisted processed data. Bump when the layout or the
#: meaning of any stored artefact changes.
PROCESSING_VERSION = "v4"

type SplitStrategy = Literal["temporal", "random"]


@dataclass(frozen=True)
class CommunityConfig:
    """Budgets and thresholds for the connected-community selection."""

    max_users: int | None = None
    max_interactions: int | None = None
    max_items: int | None = None
    seed: int = settings.SELECTION_SEED
    fields: list[str] | None = None
    min_history_retention: float = 0.7

    def __post_init__(self) -> None:
        for name, value in (
            ("max_users", self.max_users),
            ("max_interactions", self.max_interactions),
            ("max_items", self.max_items),
        ):
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 1
            ):
                raise ValueError(f"{name} must be a positive integer or None.")
        if not 0 < self.min_history_retention <= 1:
            raise ValueError("min_history_retention must be in (0, 1].")
        if self.fields is not None:
            object.__setattr__(self, "fields", sorted(set(self.fields)))

    def metadata(self, min_interactions: int) -> dict[str, Any]:
        """Report-friendly representation shared by selection and cache identity."""
        return {
            "policy": "community",
            "max_users": self.max_users,
            "max_interactions": self.max_interactions,
            "max_items": self.max_items,
            "seed": self.seed,
            "community_fields": self.fields,
            "min_interactions": min_interactions,
            "min_history_retention": self.min_history_retention,
        }


@dataclass(frozen=True)
class DataConfig:
    """Every option that affects the processed data and its cache."""

    dataset: DatasetName
    test_ratio: float = settings.TEST_RATIO
    val_ratio: float = settings.VAL_RATIO
    min_interactions: int = settings.MIN_INTERACTIONS
    remove_sparse: bool = settings.REMOVE_SPARSE
    deduplicate: bool = settings.DEDUPLICATE_INTERACTIONS
    random_state: int | None = None
    community: CommunityConfig = field(default_factory=CommunityConfig)

    def __post_init__(self) -> None:
        if not 0 < self.test_ratio < 1 or not 0 < self.val_ratio < 1:
            raise ValueError("Validation/test ratios must be in (0, 1).")
        if self.test_ratio + self.val_ratio >= 1:
            raise ValueError("Validation/test ratios must sum to less than 1.")
        if self.min_interactions < 3:
            raise ValueError(
                "min_interactions must be at least 3 to create train/val/test splits."
            )

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
