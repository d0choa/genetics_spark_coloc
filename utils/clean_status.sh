#!/usr/bin/env bash
REQUESTED_REF=$1

CURRENT_REF=$(git rev-parse --abbrev-ref HEAD)

if [ "$REQUESTED_REF" != "$CURRENT_REF" ]; then
    echo "Requested branch $REQUESTED_REF is not the current branch $CURRENT_REF, skipping status checks"
    exit 0
fi

echo "Fetching version changes..."
git fetch
if output=$(git status --porcelain) && [[ -n "$output" ]]; then
    exit 1
fi
echo "Checking if branch is up-to-date with remote..."
if output=$(git push --dry-run --porcelain) && [[ "$output" != *"[up to date]"* ]]; then
    exit 1
fi
