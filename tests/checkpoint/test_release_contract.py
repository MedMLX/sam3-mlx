from sam3_mlx.convert import DEFAULT_MLX_CHECKPOINT
from sam3_mlx.release_contract import (
    MLX_CHECKPOINT_REPO,
    MLX_CHECKPOINT_REVISION,
    RELEASE_IMAGES,
    build_oracle_bindings,
)


def test_release_contract_matches_default_hub_checkpoint_pin() -> None:
    assert DEFAULT_MLX_CHECKPOINT.repo == MLX_CHECKPOINT_REPO
    assert DEFAULT_MLX_CHECKPOINT.revision == MLX_CHECKPOINT_REVISION


def test_oracle_bindings_name_the_image_and_case_spec() -> None:
    bindings = build_oracle_bindings(
        image="official-checkout/assets/images/test_image.jpg",
        case_spec="/tmp/run/cases.json",
        confidence_threshold=0.5,
    )

    assert bindings["image"] == "test_image.jpg"
    assert bindings["case_spec"] == "cases.json"
    assert bindings["confidence_threshold"] == 0.5


def test_release_images_are_distinct() -> None:
    assert set(RELEASE_IMAGES) == {"example", "holdout"}
    assert RELEASE_IMAGES["example"]["path"] != RELEASE_IMAGES["holdout"]["path"]
    for image in RELEASE_IMAGES.values():
        assert image["path"].startswith("official-checkout/assets/images/")
        assert len(image["size"]) == 2
