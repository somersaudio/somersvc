#!/bin/bash
# Build a drag-to-Applications DMG from dist/SomerSVC.app
# Usage: ./build_dmg.sh

set -e

APP="dist/SomerSVC.app"
[ -d "$APP" ] || { echo "ERROR: $APP not found. Run pyinstaller --noconfirm somersvc.spec first."; exit 1; }

# Pull version from Info.plist
VERSION=$(/usr/libexec/PlistBuddy -c "Print :CFBundleShortVersionString" "$APP/Contents/Info.plist" 2>/dev/null || echo "0.0.0")
ARCH=$(uname -m)
DMG_NAME="SomerSVC-v${VERSION}-${ARCH}.dmg"

# Clear any leftover staging dir
STAGING=$(mktemp -d)
trap "rm -rf $STAGING" EXIT

# Lay out the DMG contents: just the .app + a shortcut to /Applications
cp -R "$APP" "$STAGING/"
ln -s /Applications "$STAGING/Applications"

# Build the dmg
rm -f "dist/$DMG_NAME"
hdiutil create -volname "SomerSVC" \
    -srcfolder "$STAGING" \
    -ov -format UDZO \
    "dist/$DMG_NAME"

echo "Built: dist/$DMG_NAME"
ls -lh "dist/$DMG_NAME"
