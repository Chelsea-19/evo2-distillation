from pathlib import Path


def test_no_windows_drive_path_in_active_source() -> None:
    root = Path(__file__).resolve().parents[1]
    active = list((root / "src").rglob("*.py")) + list((root / "scripts").glob("*.py"))
    for path in active:
        text = path.read_text(encoding="utf-8")
        assert "C:\\\\Users" not in text
        assert "D:\\\\BMDSIS" not in text

