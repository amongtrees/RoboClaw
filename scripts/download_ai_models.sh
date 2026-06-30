#!/bin/bash
# Download open-source AI model weights for RoboClaw Phase 2.
#
# Downloads three model types:
#   VLA:         OpenVLA-7B (Vision-Language-Action, ~15 GB)
#   VLN:         InternVLA-N1 DualVLN (Vision-Language Navigation, ~20 GB)
#   World Model: Cosmos-Predict2.5 2B (NVIDIA, ~5 GB)
#
# Prerequisites:
#   pip install huggingface_hub transformers torch accelerate
#
# Usage:
#   bash scripts/download_ai_models.sh              # all models
#   bash scripts/download_ai_models.sh vla vln      # specific models
#
# Default directory: models/ai/
# Override: AI_MODEL_DIR=/path bash scripts/download_ai_models.sh

set -euo pipefail

MODEL_DIR="${AI_MODEL_DIR:-models/ai}"
mkdir -p "$MODEL_DIR"

# ---------------------------------------------------------------------------
# VLA: OpenVLA-7B
# ---------------------------------------------------------------------------
download_vla() {
    local repo="openvla/openvla-7b"
    local out="$MODEL_DIR/openvla-7b"
    echo "━━━ [VLA] OpenVLA-7B ━━━"
    if [ -d "$out/config.json" ] 2>/dev/null || [ -f "$out/config.json" ]; then
        echo "  Already downloaded: $out"
        return
    fi
    echo "  Downloading from HuggingFace: $repo"
    echo "  Target: $out"
    echo "  Size: ~15 GB, may take 10-20 minutes..."
    python -c "
from huggingface_hub import snapshot_download
snapshot_download('$repo', local_dir='$out',
                  ignore_patterns=['*.msgpack', '*.h5'])
print('OpenVLA-7B downloaded.')
" || {
        echo "  huggingface_hub failed, trying CLI..."
        huggingface-cli download "$repo" --local-dir "$out"
    }
    echo "  ✓ VLA done."
}

# ---------------------------------------------------------------------------
# VLN: InternVLA-N1 DualVLN
# ---------------------------------------------------------------------------
download_vln() {
    local repo="InternRobotics/InternVLA-N1-DualVLN"
    local out="$MODEL_DIR/internvla-n1-dualvln"
    echo "━━━ [VLN] InternVLA-N1 DualVLN ━━━"
    if [ -f "$out/config.json" ]; then
        echo "  Already downloaded: $out"
        return
    fi
    echo "  Downloading from HuggingFace: $repo"
    echo "  Target: $out"
    echo "  Size: ~20 GB, may take 15-30 minutes..."
    python -c "
from huggingface_hub import snapshot_download
snapshot_download('$repo', local_dir='$out')
print('InternVLA-N1 DualVLN downloaded.')
" || {
        huggingface-cli download "$repo" --local-dir "$out"
    }
    echo "  ✓ VLN done."
}

# ---------------------------------------------------------------------------
# World Model: Cosmos-Predict2.5 2B
# ---------------------------------------------------------------------------
download_world_model() {
    local repo="nvidia-cosmos/cosmos-predict2.5-2b"
    local out="$MODEL_DIR/cosmos-predict2.5-2b"
    echo "━━━ [World Model] Cosmos-Predict2.5 2B ━━━"
    if [ -f "$out/config.json" ]; then
        echo "  Already downloaded: $out"
        return
    fi
    echo "  Downloading from HuggingFace: $repo"
    echo "  Target: $out"
    echo "  Size: ~5 GB, may take 5-10 minutes..."
    python -c "
from huggingface_hub import snapshot_download
snapshot_download('$repo', local_dir='$out')
print('Cosmos-Predict2.5 2B downloaded.')
" || {
        huggingface-cli download "$repo" --local-dir "$out"
    }
    echo "  ✓ World Model done."
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
TARGETS=("${@:-vla vln world_model}")
for t in "${TARGETS[@]}"; do
    case "$t" in
        vla)         download_vla ;;
        vln)         download_vln ;;
        world_model) download_world_model ;;
        all)
            download_vla
            download_vln
            download_world_model
            break
            ;;
        *) echo "Unknown target: $t (use: vla vln world_model all)" ;;
    esac
done

echo ""
echo "══════════════════════════════════════════"
echo "Downloads complete. Models in: $MODEL_DIR"
echo ""
echo "Next — launch inference servers:"
echo "  python scripts/serve_vla.py"
echo "  python scripts/serve_vln.py"
echo "  python scripts/serve_world_model.py"
echo ""
echo "Then update configs/models/*.yaml base_url → http://localhost:8001 (VLA)"
echo "                                            → http://localhost:8002 (VLN)"
echo "                                            → http://localhost:8003 (World)"
echo "══════════════════════════════════════════"
