#!/usr/bin/env bash
set -euo pipefail
python -m pip install --upgrade pip
python -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.5.1 torchvision==0.20.1
python -m pip install --no-cache-dir -r requirements-cloud.txt
python scripts/verify_model.py
