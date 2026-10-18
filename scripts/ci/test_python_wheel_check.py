"""The wheel check must reject a wheel whose machine code is not the architecture its name claims."""

import struct
import subprocess
import sys
import zipfile
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent / "check_python_wheel.py"
X86_64 = 0x01000007
ARM64 = 0x0100000C
LINUX_ELF = b"\x7fELF" + bytes(60)


def thin(cpu: int) -> bytes:
    """A 64-bit little-endian Mach-O header, padded the way a real file is longer than its header."""
    return struct.pack("<IIIIIII", 0xFEEDFACF, cpu, 3, 6, 0, 0, 0) + bytes(64)


def fat(*cpus: int) -> bytes:
    header = struct.pack(">II", 0xCAFEBABE, len(cpus))
    entries = b"".join(struct.pack(">IIIII", cpu, 3, 4096 * (i + 1), 64, 12) for i, cpu in enumerate(cpus))
    return header + entries + bytes(64)


def make_wheel(directory: Path, platform: str, members: dict[str, bytes]) -> Path:
    wheel = directory / f"liter_llm-2.2.5-cp310-abi3-{platform}.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        for name, data in members.items():
            archive.writestr(name, data)
    return wheel


def run(directory: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), str(directory), *args], capture_output=True, text=True, check=False
    )


def test_x86_64_wheel_with_x86_64_code_passes(tmp_path: Path) -> None:
    make_wheel(
        tmp_path,
        "macosx_11_0_x86_64",
        {"liter_llm/_internal_bindings.abi3.so": thin(X86_64), "liter_llm/__init__.py": b""},
    )
    result = run(tmp_path, r"macosx_\d+_\d+_x86_64", "x86_64")
    assert result.returncode == 0, result.stdout + result.stderr


def test_x86_64_tag_over_arm64_code_fails(tmp_path: Path) -> None:
    """The failure this check exists for: the right name over the wrong machine code."""
    make_wheel(tmp_path, "macosx_11_0_x86_64", {"liter_llm/_internal_bindings.abi3.so": thin(ARM64)})
    result = run(tmp_path, r"macosx_\d+_\d+_x86_64", "x86_64")
    assert result.returncode == 1
    assert "has architectures ['arm64']" in result.stdout


def test_one_wrong_file_among_right_files_fails(tmp_path: Path) -> None:
    make_wheel(tmp_path, "macosx_11_0_x86_64", {"a.so": thin(X86_64), "b.dylib": thin(ARM64)})
    result = run(tmp_path, r"macosx_\d+_\d+_x86_64", "x86_64")
    assert result.returncode == 1
    assert "b.dylib" in result.stdout
    assert "a.so has" not in result.stdout


def test_universal_file_fails_because_it_holds_a_second_architecture(tmp_path: Path) -> None:
    make_wheel(tmp_path, "macosx_11_0_x86_64", {"a.so": fat(X86_64, ARM64)})
    result = run(tmp_path, r"macosx_\d+_\d+_x86_64", "x86_64")
    assert result.returncode == 1
    assert "['x86_64', 'arm64']" in result.stdout


def test_single_architecture_fat_file_passes(tmp_path: Path) -> None:
    make_wheel(tmp_path, "macosx_11_0_x86_64", {"a.so": fat(X86_64)})
    assert run(tmp_path, r"macosx_\d+_\d+_x86_64", "x86_64").returncode == 0


def test_wrong_platform_tag_fails_even_with_right_code(tmp_path: Path) -> None:
    make_wheel(tmp_path, "macosx_11_0_arm64", {"a.so": thin(X86_64)})
    result = run(tmp_path, r"macosx_\d+_\d+_x86_64", "x86_64")
    assert result.returncode == 1
    assert "platform tag 'macosx_11_0_arm64'" in result.stdout


def test_wheel_with_no_mach_o_file_fails(tmp_path: Path) -> None:
    make_wheel(tmp_path, "macosx_11_0_x86_64", {"liter_llm/__init__.py": b"", "a.so": LINUX_ELF})
    result = run(tmp_path, r"macosx_\d+_\d+_x86_64", "x86_64")
    assert result.returncode == 1
    assert "no Mach-O file" in result.stdout


def test_empty_directory_fails(tmp_path: Path) -> None:
    assert run(tmp_path, r".*", "x86_64").returncode == 1


def test_tag_only_check_ignores_machine_code(tmp_path: Path) -> None:
    make_wheel(tmp_path, "manylinux_2_17_x86_64", {"a.so": LINUX_ELF})
    assert run(tmp_path, r"manylinux.*_x86_64").returncode == 0
    assert run(tmp_path, r"manylinux.*_aarch64").returncode == 1


def test_empty_architecture_argument_reads_tags_only(tmp_path: Path) -> None:
    make_wheel(tmp_path, "win_amd64", {"a.pyd": LINUX_ELF})
    assert run(tmp_path, "win_amd64", "").returncode == 0
