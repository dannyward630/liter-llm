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


ROOT = Path(__file__).resolve().parents[2]


def wheel_job() -> dict:
    return workflow("python-wheels.yaml")["jobs"]["build"]


def workflow(name: str) -> dict:
    import yaml

    return yaml.safe_load((ROOT / ".github/workflows" / name).read_text())


def test_release_builds_an_intel_macos_wheel_on_the_arm64_runner() -> None:
    legs = {leg["label"]: leg for leg in wheel_job()["strategy"]["matrix"]["include"]}
    intel = legs["macos-x86_64"]
    assert intel["runner"] == "macos-latest"
    assert intel["cibw_archs_macos"] == "x86_64"
    assert intel["wheel_macho_arch"] == "x86_64"
    assert intel["wheel_platform"] == "macosx_11_0_x86_64"
    assert "rustup target add x86_64-apple-darwin" in intel["before_build_macos"]
    build = next(step for step in wheel_job()["steps"] if "build-python-wheels" in step.get("uses", ""))
    assert "macosx_x86_64" not in build["with"]["cibw-skip"]
    assert "MACOSX_DEPLOYMENT_TARGET=11.0" in build["with"]["cibw-environment-macos"]


def test_every_wheel_is_checked_before_it_is_uploaded() -> None:
    steps = wheel_job()["steps"]
    check = next(i for i, step in enumerate(steps) if "check_python_wheel.py" in step.get("run", ""))
    upload = next(i for i, step in enumerate(steps) if "upload-artifact" in step.get("uses", ""))
    build = next(i for i, step in enumerate(steps) if "build-python-wheels" in step.get("uses", ""))
    assert build < check < upload
    assert steps[build]["with"]["upload-artifact"] == "false"
    assert steps[upload]["with"]["name"] == "python-wheel-${{ matrix.label }}"


def test_release_and_pull_request_check_build_wheels_with_the_same_workflow() -> None:
    release = workflow("publish.yaml")["jobs"]["build-python-wheels"]
    check = workflow("ci-python-wheels.yaml")["jobs"]["wheels"]
    assert release["uses"] == "./.github/workflows/python-wheels.yaml"
    assert check["uses"] == "./.github/workflows/python-wheels.yaml"


def test_pull_request_wheel_check_has_no_path_to_a_registry() -> None:
    check = workflow("ci-python-wheels.yaml")
    assert "pull_request" in check[True]
    assert "secrets" not in check["jobs"]["wheels"]
    assert check["permissions"] == {"contents": "read"}
    reusable = (ROOT / ".github/workflows/python-wheels.yaml").read_text()
    for word in ("secrets.", "id-token", "twine", "pypa/gh-action-pypi-publish", "maturin publish"):
        assert word not in reusable


RELEASE_WHEEL_LEGS = {
    "linux-x86_64": ("ubuntu-latest", "auto", "manylinux.*_x86_64", None),
    "linux-aarch64": ("ubuntu-24.04-arm", "auto", "manylinux.*_aarch64", None),
    "macos-arm64": ("macos-latest", "arm64", "macosx_11_0_arm64", "arm64"),
    "macos-x86_64": ("macos-latest", "x86_64", "macosx_11_0_x86_64", "x86_64"),
    "windows-x86_64": ("windows-latest", "auto", "win_amd64", None),
}


def test_release_job_builds_all_five_wheels_for_the_tag_and_uploads_them() -> None:
    release = workflow("publish.yaml")["jobs"]["build-python-wheels"]
    assert release["uses"] == "./.github/workflows/python-wheels.yaml"
    assert release["needs"] == ["prepare", "registry-results"]
    assert release["with"] == {"checkout_ref": "${{ needs.prepare.outputs.checkout_ref }}"}
    assert release["if"] == (
        "${{ !inputs.plugins-only && (always() && needs.prepare.outputs.release_python == 'true'"
        " && (needs.registry-results.outputs.pypi != 'true' || needs.prepare.outputs.force_republish == 'true')) }}"
    )

    job = wheel_job()
    legs = {
        leg["label"]: (leg["runner"], leg["cibw_archs_macos"], leg["wheel_platform"], leg.get("wheel_macho_arch"))
        for leg in job["strategy"]["matrix"]["include"]
    }
    assert legs == RELEASE_WHEEL_LEGS
    assert job["env"]["CIBW_ARCHS_MACOS"] == "${{ matrix.cibw_archs_macos }}"

    checkout = next(step for step in job["steps"] if "actions/checkout" in step.get("uses", ""))
    assert checkout["with"] == {"ref": "${{ inputs.checkout_ref }}"}
    upload = next(step for step in job["steps"] if "upload-artifact" in step.get("uses", ""))
    assert upload["with"]["path"] == "./wheelhouse/*.whl"


def test_pull_request_wheel_check_watches_the_crates_the_wheel_is_built_from() -> None:
    paths = workflow("ci-python-wheels.yaml")[True]["pull_request"]["paths"]
    for watched in ("crates/liter-llm/**", "crates/liter-llm-py/**", "Cargo.toml", "Cargo.lock"):
        assert watched in paths
