"""Entry point for the AutoPodcast graphical interface."""

from __future__ import annotations


def main() -> None:
    """Launch the AutoPodcast GUI window."""
    from autopodcast.gui.app import AutoPodcastApp

    app = AutoPodcastApp()
    app.mainloop()


if __name__ == "__main__":
    main()
