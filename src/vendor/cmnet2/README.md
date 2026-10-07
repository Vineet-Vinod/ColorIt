Vendored CMNET2 inference subset.

Source: https://github.com/dan64/cmnet2
Revision: e0d51432d224476769babfbff5e90f531a454939
Author: Dan64. Based on ColorMNet by Yixin Yang, Jiangxin Dong, Jinhui Tang,
and Jinshan Pan, with code from XMem and XMem++.

Only modules needed by ColorMNetRender are included. Training scripts,
datasets, demos, and model weights are excluded. Internal imports use the
`src.vendor.cmnet2` package.

Local adaptations replace CUDA rendering operations with Apple Silicon MPS,
disable the CUDA correlation extension and fix its attention fallback shapes,
disable redundant ResNet weight downloads, and reject incomplete checkpoints.
The renderer accepts explicit checkpoint and DINOv3 backbone paths, so no
generated models.json, source archive, or installation patch is needed.

CMNET2 inherits ColorMNet's CC BY-NC-SA 4.0 license and component-specific
terms. This subset and its local adaptations retain those terms. The original
ColorMNet license text and component notices are retained in LICENSES, copied
from https://github.com/yyang181/colormnet/blob/main/LICENSES. CMNET2's original
README and credits are retained in README.upstream.md.
