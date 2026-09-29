#!/usr/bin/env bash
set -euo pipefail

repository_directory="$(
    cd "$(dirname "${BASH_SOURCE[0]}")/.."
    pwd
)"

cd "$repository_directory/python/src/robot_arm/"
uv run python -m apps.camera_view
cd "$repository_directory"
find . -type f -name '*.py[co]' -delete -o -type d -name __pycache__ -delete  