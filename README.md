# sam3-mlx

Segment images from text and geometric prompts with an unofficial native MLX port of SAM 3.

[Docs](docs/usage.md) · [MedMLX](https://github.com/MedMLX)

![Text prompts “person” and “helmet” → masks on a real photograph.](https://raw.githubusercontent.com/MedMLX/sam3-mlx/main/docs/showcase.png)

*Text prompts “person” and “helmet” → masks on a real photograph.*

| Input | Output |
| :--- | :--- |
| Image + text, boxes or points | Masks, boxes and confidence scores |

**Apple Silicon · Metal · Python 3.13+** · [uv](https://docs.astral.sh/uv/)

## Quick start

```bash
uv venv --python 3.13
source .venv/bin/activate

repo="https://github.com/MedMLX/sam3-mlx"
release="$repo/releases/download/v0.1.3"
uv pip install \
  "$release/sam3_mlx-0.1.3-py3-none-any.whl"
```

Weights download automatically; Hugging Face access to the official checkpoint is required.

```python
from sam3_mlx import build_sam3_image_model
from sam3_mlx.model.sam3_image_processor import Sam3Processor

model = build_sam3_image_model()
processor = Sam3Processor(model, resolution=1008)

state = processor.set_image("image.jpg")
state = processor.set_text_prompt("shoe", state)

masks, boxes, scores = state["masks"], state["boxes"], state["scores"]
```

Use a resolution divisible by 14. Selected-frame video runs each frame independently; temporal tracking and SAM 3.1 are experimental.

## License

SAM License; see [LICENSE](LICENSE) and [third-party notices](THIRD_PARTY_NOTICES.md). Unofficial port; not affiliated with Meta. Official checkpoints have separate access terms.

For research and education. Outputs require human review.
