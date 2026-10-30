"""校验实际构建包的完整性，避免源码测试通过而分发文件缺失。"""

from __future__ import annotations

import tarfile
import tomllib
import zipfile
from email.parser import BytesParser
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
REQUIRED_SOURCE = {
    "app.py",
    "AGENTS.md",
    "README.md",
    "LICENSE",
    "CONTRIBUTING.md",
    "CHANGELOG.md",
    "pyproject.toml",
    "uv.lock",
    ".gitignore",
    ".gitattributes",
    ".streamlit/config.toml",
    ".github/workflows/ci.yml",
    "sample_data/README.md",
    "sample_data/sample_prices.csv",
    "scripts/generate_sample_data.py",
    "scripts/verify_distributions.py",
    "docs/formulas.md",
    "docs/releasing.md",
    "运行.cmd",
}
PRIVATE_ROOTS = {"data", "uploads", "exports", "results"}
EXCLUDED_PARTS = {
    ".venv",
    ".python",
    ".cache",
    ".uv-cache",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    ".git",
}


def check_path(name: str) -> None:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"包中含有不安全路径：{name}")
    if path.parts[0] in PRIVATE_ROOTS or EXCLUDED_PARTS.intersection(path.parts):
        raise ValueError(f"包中含有本地目录：{name}")
    if path.name.startswith(".env") or path.name == "secrets.toml":
        raise ValueError(f"包中含有凭据配置：{name}")
    if path.suffix in {".log", ".pyc", ".pem", ".key"}:
        raise ValueError(f"包中含有不应分发的文件：{name}")


def require_files(expected: dict[str, bytes], actual: dict[str, bytes], label: str) -> None:
    for name, content in expected.items():
        if name not in actual:
            raise ValueError(f"{label} 缺少文件：{name}")
        if actual[name] != content:
            raise ValueError(f"{label} 中的文件与当前源码不同：{name}")


def main() -> None:
    version = tomllib.loads((ROOT / "pyproject.toml").read_text("utf-8"))["project"]["version"]
    prefix = f"portfolio_risk_lab-{version}"
    expected_python = {
        file.relative_to(ROOT).as_posix(): file.read_bytes()
        for file in (ROOT / "src" / "portfolio_risk_lab").rglob("*.py")
    }
    expected_source = {
        name: (ROOT / name).read_bytes() for name in REQUIRED_SOURCE
    } | expected_python
    expected_source.update(
        {
            file.relative_to(ROOT).as_posix(): file.read_bytes()
            for file in (ROOT / "tests").glob("*.py")
        }
    )
    source_files: dict[str, bytes] = {}
    with tarfile.open(ROOT / "dist" / f"{prefix}.tar.gz") as archive:
        for member in archive.getmembers():
            path = PurePosixPath(member.name)
            if not member.isfile() or path.parts[0] != prefix or len(path.parts) < 2:
                raise ValueError(f"无效源码包成员：{member.name}")
            name = PurePosixPath(*path.parts[1:]).as_posix()
            check_path(name)
            stream = archive.extractfile(member)
            if stream is None or name in source_files:
                raise ValueError(f"无效或重复源码包文件：{name}")
            source_files[name] = stream.read()
    require_files(expected_source, source_files, "源码包")
    with zipfile.ZipFile(ROOT / "dist" / f"{prefix}-py3-none-any.whl") as wheel:
        if wheel.testzip() is not None:
            raise ValueError("Wheel CRC 校验失败。")
        wheel_files = {name: wheel.read(name) for name in wheel.namelist()}
    for name in wheel_files:
        check_path(name)
    require_files(
        {name.removeprefix("src/"): data for name, data in expected_python.items()},
        wheel_files,
        "Wheel",
    )
    metadata = BytesParser().parsebytes(wheel_files[f"{prefix}.dist-info/METADATA"])
    if metadata["Version"] != version or metadata["License-Expression"] != "MIT":
        raise ValueError("Wheel 版本或许可证元数据不一致。")
    require_files(
        {f"{prefix}.dist-info/licenses/LICENSE": (ROOT / "LICENSE").read_bytes()},
        wheel_files,
        "Wheel",
    )
    print(
        f"Distributions verified: {len(source_files)} source files, {len(wheel_files)} wheel files."
    )


if __name__ == "__main__":
    main()
