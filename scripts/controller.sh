#!/usr/bin/env bash

set -e

repository_directory="$(
    cd "$(dirname "${BASH_SOURCE[0]}")/.."
    pwd
)"

cd "$repository_directory/python/src/robot_arm/"
exec uv run python -m apps.controller
cd "$repository_directory"
find . -type f -name '*.py[co]' -delete -o -type d -name __pycache__ -delete   