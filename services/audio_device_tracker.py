"""Keep every QAudioOutput in the app pointed at the system's *current*
default output device, even when the user switches outputs at the OS
level (e.g. plugs in AirPods, swaps to a Bluetooth speaker, picks a
different output in macOS' Sound preferences).

Qt's QMediaPlayer/QAudioOutput captures whatever device was default at
construction time and doesn't re-check, so without this the app keeps
playing through the original device.

Usage:
    from services.audio_device_tracker import register_audio_output
    out = QAudioOutput()
    register_audio_output(out)
"""

from __future__ import annotations

import weakref
from typing import Optional

from PyQt6.QtCore import QObject
from PyQt6.QtMultimedia import QAudioOutput, QMediaDevices


class _AudioDeviceTracker(QObject):
    def __init__(self):
        super().__init__()
        self._devices = QMediaDevices(self)
        self._outputs: weakref.WeakSet[QAudioOutput] = weakref.WeakSet()
        # `audioOutputsChanged` fires both when the device list changes AND
        # when the default selection changes — exactly what we want.
        self._devices.audioOutputsChanged.connect(self._on_outputs_changed)

    def register(self, output: QAudioOutput) -> None:
        """Track a QAudioOutput so it follows the system default."""
        if output is None:
            return
        # Snap to the current default immediately so newly-created outputs
        # also pick up the latest device, not whatever Qt's startup default was.
        try:
            current = QMediaDevices.defaultAudioOutput()
            if current is not None and not current.isNull():
                output.setDevice(current)
        except Exception:
            pass
        self._outputs.add(output)

    def _on_outputs_changed(self) -> None:
        new_default = QMediaDevices.defaultAudioOutput()
        if new_default is None or new_default.isNull():
            return
        for out in list(self._outputs):
            try:
                out.setDevice(new_default)
            except RuntimeError:
                # Underlying C++ object went away — WeakSet will drop it on
                # the next iteration.
                pass


_tracker: Optional[_AudioDeviceTracker] = None


def _get_tracker() -> _AudioDeviceTracker:
    global _tracker
    if _tracker is None:
        _tracker = _AudioDeviceTracker()
    return _tracker


def register_audio_output(output: QAudioOutput) -> None:
    """Make a QAudioOutput track the system default. Safe to call once
    per QAudioOutput at construction time."""
    _get_tracker().register(output)
