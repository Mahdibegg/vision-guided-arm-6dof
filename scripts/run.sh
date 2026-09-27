#!/usr/bin/env bash

set -e

# Resolve directories relative to the script's location in scripts/
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CS_DIR="$PROJECT_ROOT/simulation/CoppeliaSim_Edu_V4_8_0_rev0_Ubuntu22_04"
SCENES_DIR="$CS_DIR/scenes"
SCENE_FILE="robot_pickup.ttt"

echo "Verifying environment dependencies..."
bash "$SCRIPT_DIR/setup.sh"

echo "Checking for $SCENE_FILE..."
if [ -f "$SCENES_DIR/$SCENE_FILE" ]; then
    echo "Found $SCENE_FILE in the scenes directory."
elif [ -f "$PROJECT_ROOT/$SCENE_FILE" ]; then
    echo "Found $SCENE_FILE in project root. Moving to scenes directory..."
    mkdir -p "$SCENES_DIR"
    cp "$PROJECT_ROOT/$SCENE_FILE" "$SCENES_DIR/"
else
    echo -e "\e[31m$SCENE_FILE file not found in root or scenes directory.\e[0m"
    exit 1
fi

echo "Starting CoppeliaSim..."
cd "$CS_DIR" || exit 1

# Launch CoppeliaSim in the background and pass the scene file to open it automatically
QT_QPA_PLATFORM=xcb ./coppeliaSim.sh "$SCENES_DIR/$SCENE_FILE" &

# Allow time for the ZeroMQ server and simulation engine to fully initialize
sleep 6

echo "Starting robot arm controller..."
cd "$SCRIPT_DIR" || exit 1
./controller.sh