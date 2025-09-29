#!/bin/bash
set -euo pipefail  # stop on error, undefined variable, or pipeline failure

# trap to print the failing command
trap 'echo "Error occurred at command: $BASH_COMMAND"' ERR

OBJECTS_SCENE_DIR=data/scenes/scene_example
OUTPUT_NPZ_DIR=data/output_npz
SCENE_NAME=scene_example
Z_UP=true

python superdec/utils/ply_to_npz.py --input_path="$OBJECTS_SCENE_DIR" --scene_name="$SCENE_NAME"

python superdec/evaluate/to_npz.py checkpoints_folder="checkpoints/normalized" output_dir="$OUTPUT_NPZ_DIR" dataset=scene scene.name="$SCENE_NAME" scene.z_up="$Z_UP"

python superdec/visualization/object_visualizer.py dataset=scene split="$SCENE_NAME" npz_folder="$OUTPUT_NPZ_DIR"