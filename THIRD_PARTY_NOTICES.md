# Third-party notices

- **Pb4U / PB4U-GNet**: the native graph model infrastructure, input adapters, runner, dataset loader and physical objectives are retained under the upstream MIT license in `external/PB4U-GNet-main/LICENSE`. Its upstream attribution, including inherited HOOD attribution, is preserved.
- **SENC**: the author baseline source snapshot and its MIT license are retained at `external/SENC/`, revision `66e433f9668982aa7d79fc31fd2892c57f19e5ef`. Baseline weights and licensed datasets are not included.
- **MCO / Explicit Frictional Contact**: the author baseline source snapshot and its MIT license are retained at `external/MCO/`, revision `34e809cab655dd191536179d4a1d793a483d7ef0`. Its complete contact-operator implementation is preserved; weights and licensed datasets are not included.
- **ContourCraft**: `arcon/detector.py` retains the upstream `find_close_faces` detector adapter. Its baseline source is retained at `external/ContourCraft-main/`, with licenses at `external/ContourCraft-main/LICENSE` and `licenses/ContourCraft-MIT.txt`.
- **CCCollisions / torch-mesh-isect**: required for self-intersection detection and counted-BVH training queries. The upstream README states that the torch-mesh-isect license also applies. Their source/headers and binaries are not redistributed here; users must obtain them under the upstream terms. Source: https://github.com/Dolorousrtur/CCCollisions and https://github.com/vchoutas/torch-mesh-isect .
- **NVIDIA CUDA samples / Warp**: sample headers are used in CUDA builds, and Warp is used for signed-distance queries. These are separately installed dependencies and are not redistributed in this archive.
- **SMPL, motion and garment assets**: not included. Obtain authorized copies through their original providers and the retained upstream data instructions.

Compatibility edits to Python dataclass defaults and PyTorch extension APIs are packaging/environment adaptations, not changes to the physical updates or evaluation definitions.
