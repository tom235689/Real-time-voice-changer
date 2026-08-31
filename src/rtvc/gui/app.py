"""GUI bootstrap."""

from __future__ import annotations

import sys

from ..config import Config


def run(cfg: Config | None = None) -> int:
    from PySide6.QtWidgets import QApplication

    from .main_window import MainWindow

    app = QApplication.instance() or QApplication(sys.argv)
    window = MainWindow(cfg or Config())
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(run())
