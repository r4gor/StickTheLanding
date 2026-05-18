#!/usr/bin/env bash
# Run any command inside the eval container with the same resource limits
# used at evaluation time. Mounts the current directory at /workspace and
# forwards the host UID so created files are owned by you, not root.
#
# Examples:
#   ./docker/run-local.sh python evaluation/evaluate.py --scenario hard --headless
#   ./docker/run-local.sh python docker/benchmark.py
#   ./docker/run-local.sh pytest tests/
#   ./docker/run-local.sh bash                                  # drop into shell
#
# Override the image tag with $CHALLENGE_IMAGE; defaults to challenge-eval.
set -euo pipefail

IMAGE="${CHALLENGE_IMAGE:-challenge-eval:base}"
CPUS="${CHALLENGE_CPUS:-6}"
MEMORY="${CHALLENGE_MEMORY:-8g}"

# Default tag is `challenge-eval:base` (built locally from
# docker/Dockerfile.eval). To run against the organizer-published
# image with the reference simulator baked in, pull and re-tag:
#   docker pull ghcr.io/skyeusoftware/catch-the-boat:2026-hackathon
#   docker tag  ghcr.io/skyeusoftware/catch-the-boat:2026-hackathon challenge-eval:full
# Then run with CHALLENGE_IMAGE=challenge-eval:full ./docker/run-local.sh ...
# The published image is public — no login required.

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    if [[ "$IMAGE" == *":full"* ]]; then
        echo "error: '$IMAGE' not found locally." >&2
        echo "Pull the organizer image first:" >&2
        echo "  docker pull --platform linux/amd64 ghcr.io/skyeusoftware/catch-the-boat:2026-hackathon" >&2
        echo "  docker tag ghcr.io/skyeusoftware/catch-the-boat:2026-hackathon challenge-eval:full" >&2
        exit 1
    fi
    echo "Image '$IMAGE' not found locally. Building from docker/Dockerfile.eval ..." >&2
    docker build --load -f docker/Dockerfile.eval -t "$IMAGE" .
fi

# Use amd64 emulation only for the organizer :full image (no ARM64 manifest).
# The locally-built :base image is native ARM64 and must NOT force amd64.
# Intentionally unquoted below so empty string adds no argument.
PLATFORM_FLAG=""
if [[ "$IMAGE" == *":full"* ]]; then
    PLATFORM_FLAG="--platform linux/amd64"
fi

# --network host: gives the container the host's network so participants
# who pull artifacts from the internet during dev don't hit DNS issues.
# Drop it if your eval policy forbids network at scoring time.
exec docker run --rm -it \
    $PLATFORM_FLAG \
    --cpus="$CPUS" \
    --memory="$MEMORY" \
    --user "$(id -u):$(id -g)" \
    -v "$(pwd)":/workspace \
    -w /workspace \
    "$IMAGE" "$@"
