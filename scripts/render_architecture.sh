#!/usr/bin/env bash
# Rebuild docs/architecture/bindra_architecture.{png,svg} from the .mmd.
#
# This exists because the committed PNG went stale: the .mmd was rewritten and
# nothing recorded how to recompile it, so the figure in the repository showed
# an architecture that no longer existed. Run this after every edit to the .mmd.
#
#   bash scripts/render_architecture.sh
#
# Needs node (for npx); mermaid-cli and its headless Chromium are fetched on
# first run and cached after that.
#
# -s 3 puts the PNG at 2352 x 2991, which is 277 DPI across a letter page.
# The figure is laid out to letter proportions on purpose: 1.272 against
# letter portrait's 1.294.
set -euo pipefail

VERSION="@mermaid-js/mermaid-cli@11.4.2"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE="$ROOT/docs/architecture/bindra_architecture.mmd"

echo "rendering $SOURCE"
npx -y "$VERSION" -i "$SOURCE" -o "${SOURCE%.mmd}.png" -b white -s 3
npx -y "$VERSION" -i "$SOURCE" -o "${SOURCE%.mmd}.svg" -b white
echo "wrote ${SOURCE%.mmd}.png and ${SOURCE%.mmd}.svg"
