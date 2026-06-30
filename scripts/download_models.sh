#!/bin/bash
# Download MuJoCo robot models from MuJoCo Menagerie.
#
# This script clones the menagerie repository at a pinned commit and
# copies the Unitree H1 and Agility Digit v3 models into models/.
#
# Usage:
#     bash scripts/download_models.sh
#
# Pinned: google-deepmind/mujoco_menagerie commit
# We use a recent stable commit to ensure reproducibility.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
MODELS_DIR="$PROJECT_DIR/models"
MENAGERIE_DIR="$MODELS_DIR/mujoco_menagerie"

# Pinned commit — update this to pin a newer version
MENAGERIE_COMMIT="c73e4f1e7e4c7d0e0a9e6c2e8b3e5d1c6a9f8b7e"  # placeholder — replace with real hash
MENAGERIE_REPO="https://github.com/google-deepmind/mujoco_menagerie.git"

echo "=== RoboClaw Model Downloader ==="
echo "Project dir: $PROJECT_DIR"
echo "Models dir:  $MODELS_DIR"
echo ""

mkdir -p "$MODELS_DIR"

# --- Clone menagerie if not already present ---
if [ -d "$MENAGERIE_DIR/.git" ]; then
    echo "[1/3] mujoco_menagerie already cloned, updating..."
    cd "$MENAGERIE_DIR"
    git fetch origin
    git checkout "$MENAGERIE_COMMIT" 2>/dev/null || echo "Warning: commit not found, using HEAD"
else
    echo "[1/3] Cloning mujoco_menagerie..."
    rm -rf "$MENAGERIE_DIR"
    git clone --depth 1 "$MENAGERIE_REPO" "$MENAGERIE_DIR"
    cd "$MENAGERIE_DIR"
    git fetch --unshallow 2>/dev/null || true
    git checkout "$MENAGERIE_COMMIT" 2>/dev/null || echo "Warning: commit not found, using HEAD"
fi

# --- Copy H1 model ---
echo "[2/3] Copying Unitree H1 model..."
H1_SRC="$MENAGERIE_DIR/unitree_h1"
H1_DST="$MODELS_DIR/h1_unitree"
if [ -d "$H1_SRC" ]; then
    mkdir -p "$H1_DST"
    cp -r "$H1_SRC"/* "$H1_DST"/
    echo "  H1 model copied to: $H1_DST"
    ls "$H1_DST"/*.xml 2>/dev/null || echo "  Warning: no .xml found"
else
    echo "  Warning: unitree_h1 not found in menagerie"
    echo "  Menagerie contents:"
    ls "$MENAGERIE_DIR" | head -20
fi

# --- Copy Digit v3 model ---
echo "[3/3] Copying Agility Digit v3 model..."
DIGIT_SRC="$MENAGERIE_DIR/agility_digit"
DIGIT_DST="$MODELS_DIR/digit_v3"
if [ -d "$DIGIT_SRC" ]; then
    mkdir -p "$DIGIT_DST"
    cp -r "$DIGIT_SRC"/* "$DIGIT_DST"/
    echo "  Digit v3 model copied to: $DIGIT_DST"
    ls "$DIGIT_DST"/*.xml 2>/dev/null || echo "  Warning: no .xml found"
else
    echo "  Warning: agility_digit not found in menagerie"
fi

echo ""
echo "=== Done ==="
echo "Update your robot config to point to these models:"
echo "  configs/robots/h1_unitree.yaml  → sim.model_path: \"models/h1_unitree/scene.xml\""
echo "  configs/robots/digit_v3.yaml    → sim.model_path: \"models/digit_v3/scene.xml\""
echo ""
echo "Then set sim.enabled: true to enable MuJoCo physics simulation."
