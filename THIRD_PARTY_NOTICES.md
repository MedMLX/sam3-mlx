# Third-party notices

The SAM License in `LICENSE` governs the SAM components. It does not replace
the retained licenses of third-party components below.

## Text tokenizer

`sam3_mlx/model/tokenizer_ve.py` adapts Meta SAM 3's tokenizer, whose header
identifies VE, open_clip and OpenAI CLIP ancestry. The tokenizer header at the
project's official source reference
[`2814fa619404a722d03e9a012e083e4f293a4e53`](https://github.com/facebookresearch/sam3/blob/2814fa619404a722d03e9a012e083e4f293a4e53/sam3/model/tokenizer_ve.py)
records the same ancestry. This identifies the inspected SAM reference; it does
not establish the historical CLIP/open_clip revision from which VE copied code.

Upstream copyright and MIT permission notices are retained in full:

- [OpenAI CLIP](#openai-clip-license), from
  <https://github.com/openai/CLIP/blob/main/LICENSE>.
- [open_clip](#openclip-license), from
  <https://github.com/mlfoundations/open_clip/blob/main/LICENSE>.

These notices accompany source and binary distributions alongside the SAM terms.
Checkpoint weights retain their separate publisher terms and are not included.

## DART tracker

`sam3_mlx/tracking.py` ports the ByteTrack tracker in DART by Mehmet Kerem
Turkcan, "Detect Anything in Real Time: From Single-Prompt Segmentation to
Multi-Class Detection" (arXiv:2603.11441), from
[`sam3/tracking/byte_tracker.py` at `6a32cab`](https://github.com/mkturkcan/DART/blob/6a32cab77784c0d8786ca29332fefcf74374e99f/sam3/tracking/byte_tracker.py).
DART is distributed under the SAM License, the same terms as this package
(`LICENSE`). The tracking method is ByteTrack: Zhang et al., "ByteTrack:
Multi-Object Tracking by Associating Every Detection Box", ECCV 2022.


<a id="openai-clip-license"></a>

## Full license: OpenAI-CLIP-MIT.txt

MIT License

Copyright (c) 2021 OpenAI

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.



<a id="openclip-license"></a>

## Full license: OpenCLIP-MIT.txt

Copyright (c) 2012-2021 Gabriel Ilharco, Mitchell Wortsman,
Nicholas Carlini, Rohan Taori, Achal Dave, Vaishaal Shankar,
John Miller, Hongseok Namkoong, Hannaneh Hajishirzi, Ali Farhadi,
Ludwig Schmidt

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

## Showcase image

The photograph of astronaut Eileen Collins is by NASA, public domain,
provided by [scikit-image](https://scikit-image.org/docs/0.24.x/api/skimage.data.html#skimage.data.astronaut).
The showcase adds SAM 3 masks.
