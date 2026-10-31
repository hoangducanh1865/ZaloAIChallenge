#!/usr/bin/env bash
# Install Kimi-Audio's inference code (kimia_infer), which lifemate/models/kimi_audio.py imports. Not on PyPI.
#
# Author: Duc Anh Hoang
set -euo pipefail
DIR=${1:-/content/Kimi-Audio}
[[ -d $DIR ]] || git clone -q --recursive https://github.com/MoonshotAI/Kimi-Audio.git "$DIR"
pip install -q -r "$DIR/requirements.txt"
pip install -q -e "$DIR"
