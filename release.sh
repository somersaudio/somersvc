#!/bin/bash
# Cut a new release: bump version, tag, push.
# GitHub Actions will build the .dmg and publish to somersvc-releases.
# Usage:  ./release.sh patch       (1.0.0 -> 1.0.1)
#         ./release.sh minor       (1.0.0 -> 1.1.0)
#         ./release.sh major       (1.0.0 -> 2.0.0)
#         ./release.sh 1.2.3       (explicit version)

set -e

cd "$(dirname "$0")"

BUMP="${1:-patch}"

# Read current version from somersvc.spec (CFBundleShortVersionString)
CURRENT=$(grep -E "CFBundleShortVersionString.*'[0-9]" somersvc.spec | head -1 | sed -E "s/.*'([0-9.]+)'.*/\1/")
[ -z "$CURRENT" ] && CURRENT="0.0.0"

case "$BUMP" in
    patch)
        IFS='.' read -r MAJ MIN PAT <<< "$CURRENT"
        NEW="$MAJ.$MIN.$((PAT + 1))"
        ;;
    minor)
        IFS='.' read -r MAJ MIN PAT <<< "$CURRENT"
        NEW="$MAJ.$((MIN + 1)).0"
        ;;
    major)
        IFS='.' read -r MAJ MIN PAT <<< "$CURRENT"
        NEW="$((MAJ + 1)).0.0"
        ;;
    *)
        # Treat as explicit version like 1.2.3
        NEW="$BUMP"
        ;;
esac

echo "Releasing v$NEW (was v$CURRENT)"
read -p "Continue? [y/N] " ok
[[ "$ok" =~ ^[Yy]$ ]] || { echo "Cancelled."; exit 1; }

# Bump version in somersvc.spec for both fields
sed -i '' -E "s|'CFBundleShortVersionString': '[^']+'|'CFBundleShortVersionString': '$NEW'|" somersvc.spec
sed -i '' -E "s|'CFBundleVersion': '[^']+'|'CFBundleVersion': '$NEW'|" somersvc.spec

git add somersvc.spec
git commit -m "Release v$NEW"
git tag "v$NEW"
git push origin main
git push origin "v$NEW"

echo ""
echo "Tag v$NEW pushed. GitHub Actions is now building the .dmg."
echo "Watch at: https://github.com/somersaudio/somersvc/actions"
