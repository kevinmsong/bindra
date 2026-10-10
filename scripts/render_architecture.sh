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
# -s 3 puts the PNG at 2352 x 2838, ratio 1.207 against letter portrait's
# 1.294, so it fits a letter page with a little vertical margin.
#
# WIDTH IS FIXED by mermaid-cli's viewport: every render comes out 2352 px
# wide whatever the settings, so the only lever on the ratio is height, and
# the only lever on height is how much goes side by side.
#
# Two things were tried and rejected. Cutting rankSpacing/padding hits the
# ratio but the subgraph titles then render struck through by the boxes
# beneath them (SHARED ADAPTERS, THREE PREDICTORS, HOW ERROR IS COUNTED).
# Narrowing wrappingWidth below ~280 hits it too, but the text becomes tiny
# and wraps mid-phrase.
#
# What works is reducing how many boxes sit on one rank: GUARD was lifted out
# of AFTER, and ANALOG GENERATION was split in two. If this needs to shrink
# again, split a wide row or cut words -- not the spacing.

set -euo pipefail

VERSION="@mermaid-js/mermaid-cli@11.4.2"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE="$ROOT/docs/architecture/bindra_architecture.mmd"

echo "rendering $SOURCE"
npx -y "$VERSION" -i "$SOURCE" -o "${SOURCE%.mmd}.png" -b white -s 3
npx -y "$VERSION" -i "$SOURCE" -o "${SOURCE%.mmd}.svg" -b white
echo "wrote ${SOURCE%.mmd}.png and ${SOURCE%.mmd}.svg"
