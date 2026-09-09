"""
Main window loader for ASTRA-Track.
Seamlessly launches the full desktop UI and manages application lifecycle.
"""

from astra_track.ui.desktop_app import launch_desktop_gui


def launch_gui():
    """Launches the primary ASTRA-Track desktop interface."""
    launch_desktop_gui()


if __name__ == "__main__":
    launch_gui()
