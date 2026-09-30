#!/usr/bin/env bash
set -eu
cd -- "$(dirname -- "$0")"
exec python3 start_website.py "$@"
