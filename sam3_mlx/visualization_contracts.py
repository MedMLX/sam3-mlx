"""Array and record shapes shared by the optional visualization helpers."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from os import PathLike
from typing import TYPE_CHECKING, NotRequired, Protocol, ReadOnly, TypedDict

import numpy as np
from numpy.typing import ArrayLike, NDArray
from PIL import Image

from sam3_mlx.agent.contracts import JsonValue
from sam3_mlx.rle import CocoRle

if TYPE_CHECKING:
    from matplotlib.colors import Colormap
    from matplotlib.image import AxesImage
    from matplotlib.patches import Patch, Rectangle
    from matplotlib.collections import PathCollection
    from matplotlib.lines import Line2D
    from matplotlib.transforms import Transform
    from matplotlib.text import Text


type Array = NDArray[np.generic]
type BoolArray = NDArray[np.bool_]
type ByteArray = NDArray[np.uint8]
type FloatArray = NDArray[np.float32]
type Color = str | ArrayLike | None
type Frame = Array | Image.Image | str | PathLike[str]
type VideoFrames = Sequence[Frame] | Array
type ImageInput = Image.Image | ArrayLike
type FormattedFrameOutput = dict[int | str, ArrayLike]
type FormattedVideoOutput = dict[int, FormattedFrameOutput]


class UncompressedRle(TypedDict):
    size: Sequence[int]
    counts: Sequence[int] | NDArray[np.integer]
    area: NotRequired[int]


type Rle = CocoRle | UncompressedRle


class MaskletOutput(TypedDict):
    out_obj_ids: ReadOnly[NDArray[np.integer] | Sequence[int]]
    out_probs: ReadOnly[NDArray[np.floating] | Sequence[float | None]]
    out_boxes_xywh: ReadOnly[NDArray[np.floating] | Sequence[Sequence[float]]]
    out_binary_masks: ReadOnly[Array | Sequence[ArrayLike]]


class ListMaskletOutput(TypedDict):
    out_obj_ids: list[int]
    out_probs: list[float]
    out_boxes_xywh: list[list[float]]
    out_binary_masks: list[ByteArray]


class CocoImageInfo(TypedDict):
    height: int
    width: int


class CocoAnnotation(TypedDict, total=False):
    id: int
    image_id: int
    category_id: int
    area: float
    iscrowd: int | bool
    bbox: Sequence[float]
    score: float
    segmentation: Rle | list[list[float]]


class PanopticSegmentInfo(TypedDict, total=False):
    id: int
    category_id: int
    isthing: bool
    area: int


class ImagePrediction(TypedDict):
    scores: ReadOnly[Array | Sequence[ArrayLike]]
    masks: ReadOnly[Array | Sequence[ArrayLike]]
    boxes: ReadOnly[Array | Sequence[ArrayLike]]


class PlotAxes(Protocol):
    """Matplotlib operations used by these helpers, excluding untyped kwargs."""

    @property
    def transAxes(self) -> Transform: ...

    def imshow(self, X: ImageInput) -> AxesImage: ...

    def text(
        self, x: float, y: float, s: str, *, color: Color = None,
        weight: str = "normal", fontsize: float = 10,
        bbox: dict[str, str | float] | None = None,
        transform: Transform | None = None,
        ha: str = "left", va: str = "baseline",
        verticalalignment: str = "baseline",
    ) -> Text: ...

    def set_title(self, label: str) -> Text: ...

    def axis(self, arg: str) -> tuple[float, float, float, float]: ...

    def add_patch(self, p: Patch) -> Patch: ...

    def plot(
        self, x: float, y: float, *, marker: str, color: str, markersize: float
    ) -> list[Line2D]: ...

    def scatter(
        self, x: ArrayLike, y: ArrayLike, *, color: str, marker: str,
        s: float, edgecolor: str, linewidth: float,
    ) -> PathCollection: ...


class PlotFigure(Protocol):
    def suptitle(self, t: str, *, fontsize: float, fontweight: str) -> Text: ...


class MatplotlibPatches(Protocol):
    Rectangle: type[Rectangle]


class MatplotlibPyplot(MatplotlibPatches, Protocol):
    def figure(self, *, figsize: tuple[float, float]) -> PlotFigure: ...

    def title(self, label: str) -> Text: ...

    def imshow(self, X: ImageInput) -> AxesImage: ...

    def gca(self) -> PlotAxes: ...

    def subplots(
        self, nrows: int = 1, ncols: int = 1, *, figsize: tuple[float, float]
    ) -> tuple[PlotFigure, PlotAxes | Iterable[PlotAxes]]: ...

    def tight_layout(self) -> None: ...

    def show(self) -> None: ...

    def get_cmap(self, name: str) -> Colormap: ...


class PandasBooleanColumn(Protocol):
    def __len__(self) -> int: ...


class PandasIdentifierColumn(Protocol):
    def eq(self, value: int) -> PandasBooleanColumn: ...


class PandasILoc[RowT](Protocol):
    def __getitem__(self, index: int) -> RowT: ...


class PandasFrame[RowT](Protocol):
    @property
    def id(self) -> PandasIdentifierColumn: ...

    @property
    def video_id(self) -> PandasIdentifierColumn: ...

    @property
    def iloc(self) -> PandasILoc[RowT]: ...

    def __getitem__(self, mask: PandasBooleanColumn) -> PandasFrame[RowT]: ...

    def __len__(self) -> int: ...

    def iterrows(self) -> Iterator[tuple[object, RowT]]: ...


class AnnotationRow(Protocol):
    @property
    def segmentations(self) -> Sequence[Rle | None]: ...

    @property
    def noun_phrase(self) -> str: ...


class VideoRow(Protocol):
    @property
    def file_names(self) -> Sequence[str]: ...


class AnnotationTables(TypedDict, total=False):
    annotations: PandasFrame[AnnotationRow]
    videos: PandasFrame[VideoRow]
    info: JsonValue
    licenses: JsonValue


class PandasModule(Protocol):
    def DataFrame(self, data: JsonValue) -> PandasFrame[object]: ...
