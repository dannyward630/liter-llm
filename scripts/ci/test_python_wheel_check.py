"""The wheel check must reject a wheel whose machine code is not the architecture its name claims."""

import posixpath
import re
import shlex
import struct
import subprocess
import sys
import zipfile
from collections.abc import Iterator
from fnmatch import fnmatchcase
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


def test_macos_wheel_is_refused_without_an_architecture(tmp_path: Path) -> None:
    """A step that loses the architecture must fail, not fall back to reading the tag."""
    make_wheel(tmp_path, "macosx_11_0_x86_64", {"a.so": thin(ARM64)})
    for args in ((r"macosx_\d+_\d+_x86_64",), (r"macosx_\d+_\d+_x86_64", "")):
        result = run(tmp_path, *args)
        assert result.returncode == 2, result.stdout + result.stderr
        assert "no expected architecture" in result.stderr


def test_unknown_architecture_is_refused(tmp_path: Path) -> None:
    make_wheel(tmp_path, "macosx_11_0_x86_64", {"a.so": thin(X86_64)})
    result = run(tmp_path, r"macosx_\d+_\d+_x86_64", "x86-64")
    assert result.returncode == 2
    assert "unknown architecture" in result.stderr


def test_tag_must_match_the_whole_pattern_not_a_prefix(tmp_path: Path) -> None:
    make_wheel(tmp_path, "macosx_11_0_x86_64", {"a.so": thin(X86_64)})
    assert run(tmp_path, "macosx_11_0_x86_64", "x86_64").returncode == 0
    result = run(tmp_path, "macosx_11_0_x86", "x86_64")
    assert result.returncode == 1
    assert "does not match 'macosx_11_0_x86'" in result.stdout


def test_binary_in_the_data_directory_is_read(tmp_path: Path) -> None:
    make_wheel(
        tmp_path,
        "macosx_11_0_x86_64",
        {"liter_llm/_internal_bindings.abi3.so": thin(X86_64), "liter_llm-2.2.5.data/scripts/tool": thin(ARM64)},
    )
    result = run(tmp_path, r"macosx_\d+_\d+_x86_64", "x86_64")
    assert result.returncode == 1
    assert "liter_llm-2.2.5.data/scripts/tool has architectures ['arm64']" in result.stdout


def test_every_macos_deployment_target_needs_an_architecture(tmp_path: Path) -> None:
    for platform in ("macosx_10_12_x86_64", "macosx_14_0_arm64"):
        directory = tmp_path / platform
        directory.mkdir()
        make_wheel(directory, platform, {"a.so": thin(ARM64)})
        result = run(directory, "macosx_.*")
        assert result.returncode == 2, (platform, result.stdout + result.stderr)
        assert "no expected architecture" in result.stderr


def test_one_macos_wheel_among_linux_wheels_needs_an_architecture(tmp_path: Path) -> None:
    make_wheel(tmp_path, "manylinux_2_28_x86_64", {"a.so": LINUX_ELF})
    assert run(tmp_path, ".*").returncode == 0
    make_wheel(tmp_path, "macosx_11_0_x86_64", {"a.so": thin(ARM64)})
    result = run(tmp_path, ".*")
    assert result.returncode == 2, result.stdout + result.stderr
    assert "no expected architecture" in result.stderr


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


def test_release_and_pull_request_check_build_wheels_with_the_same_workflow() -> None:
    release = workflow("publish.yaml")["jobs"]["build-python-wheels"]
    check = workflow("ci-python-wheels.yaml")["jobs"]["wheels"]
    assert release["uses"] == "./.github/workflows/python-wheels.yaml"
    assert check["uses"] == "./.github/workflows/python-wheels.yaml"


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
    for watched in ("crates/liter-llm/**", "crates/liter-llm-py/**", "Cargo.toml", "Cargo.lock", "rust-toolchain.toml"):
        assert watched in paths


def test_pull_request_wheel_check_runs_only_in_this_repository() -> None:
    assert "github.repository == 'xberg-io/liter-llm'" in workflow("ci-python-wheels.yaml")["jobs"]["wheels"]["if"]


CONCRETE_TAGS = {
    "linux-x86_64": "manylinux_2_28_x86_64",
    "linux-aarch64": "manylinux_2_28_aarch64",
    "macos-arm64": "macosx_11_0_arm64",
    "macos-x86_64": "macosx_11_0_x86_64",
    "windows-x86_64": "win_amd64",
}


def run_check_step(leg: dict, wheel_dir: Path) -> subprocess.CompletedProcess[str]:
    """Run the check step of the wheel job as the workflow wires it, with the leg's matrix values."""
    step = next(s for s in wheel_job()["steps"] if "check_python_wheel.py" in s.get("run", ""))
    env = {}
    for name, expression in step["env"].items():
        assert expression.startswith("${{ matrix."), expression
        assert expression.endswith(" }}"), expression
        env[name] = str(leg.get(expression[len("${{ matrix.") : -len(" }}")], ""))
    bin_dir = wheel_dir / "bin"
    bin_dir.mkdir()
    (bin_dir / "python").symlink_to(sys.executable)
    (wheel_dir / "scripts").symlink_to(ROOT / "scripts")
    wheelhouse = wheel_dir / "wheelhouse"
    wheelhouse.mkdir()
    for wheel in wheel_dir.glob("*.whl"):
        wheel.rename(wheelhouse / wheel.name)
    return subprocess.run(
        ["bash", "-c", step["run"]],
        cwd=wheel_dir,
        env={"PATH": f"{bin_dir}:/usr/bin:/bin", **env},
        capture_output=True,
        text=True,
        check=False,
    )


def test_check_step_gives_each_leg_its_own_architecture(tmp_path: Path) -> None:
    """Drives the step the way the release runs it: the Intel leg must reject arm64 code."""
    legs = {leg["label"]: leg for leg in wheel_job()["strategy"]["matrix"]["include"]}
    assert set(legs) == set(CONCRETE_TAGS)
    for label, leg in legs.items():
        expected = {"x86_64": X86_64, "arm64": ARM64}.get(leg.get("wheel_macho_arch"))
        code = thin(expected) if expected else LINUX_ELF
        good = tmp_path / f"good-{label}"
        good.mkdir()
        make_wheel(good, CONCRETE_TAGS[label], {"liter_llm/_internal_bindings.abi3.so": code})
        result = run_check_step(leg, good)
        assert result.returncode == 0, (label, result.stdout + result.stderr)
        if expected:
            wrong = tmp_path / f"wrong-{label}"
            wrong.mkdir()
            other = ARM64 if expected == X86_64 else X86_64
            make_wheel(wrong, CONCRETE_TAGS[label], {"liter_llm/_internal_bindings.abi3.so": thin(other)})
            assert run_check_step(leg, wrong).returncode == 1, label


def test_wheel_build_inputs_are_pinned() -> None:
    job = wheel_job()
    build = next(s for s in job["steps"] if "build-python-wheels" in s.get("uses", ""))["with"]
    assert build["cibw-build"] == "cp310-*"
    assert "--compatibility linux" in build["cibw-environment"]
    assert (ROOT / build["package-dir"] / "pyproject.toml").is_file()
    assert job["timeout-minutes"] == 90
    upload = next(s for s in job["steps"] if "upload-artifact" in s.get("uses", ""))
    assert upload["with"]["retention-days"] == 7
    assert upload["with"]["if-no-files-found"] == "error"


# The release path rests on five invariants, one test each. Each test reads the parsed workflows and compares
# whole key sets and whole values, so any edit that breaks the invariant fails it, whatever shape the edit takes.
# A key that no test expects on a guarded job or step is an error, not a value the test ignores.

PYPI_PUBLISH_CONDITION = (
    "${{ !inputs.plugins-only && (always() && needs.prepare.outputs.release_python == 'true'"
    " && needs.validate-versions.result == 'success' && needs.build-python-wheels.result == 'success'"
    " && needs.build-python-sdist.result == 'success'"
    " && (needs.registry-results.outputs.pypi != 'true' || needs.prepare.outputs.force_republish == 'true')) }}"
)
LIST_PACKAGES_RUN = 'echo "Packages in dist:"\nls -lh dist/ 2>/dev/null || echo "No packages found"\n'


def strings(node: object) -> Iterator[str]:
    """Every key and every string value in a parsed workflow, at any depth."""
    if isinstance(node, dict):
        for key, value in node.items():
            yield str(key)
            yield from strings(value)
    elif isinstance(node, list):
        for item in node:
            yield from strings(item)
    elif isinstance(node, str):
        yield node


def step_index(steps: list[dict], uses: str) -> int:
    (index,) = (i for i, step in enumerate(steps) if uses in step.get("uses", ""))
    return index


def check_step_index(steps: list[dict]) -> int:
    (index,) = (i for i, step in enumerate(steps) if "check_python_wheel.py" in step.get("run", ""))
    return index


def publish_pypi_job() -> dict:
    return workflow("publish.yaml")["jobs"]["publish-pypi"]


def release_run_uploads() -> Iterator[tuple[str, str]]:
    """Every artifact upload in the release run as (job, artifact name), reusable workflows included."""
    for job_id, job in workflow("publish.yaml")["jobs"].items():
        jobs = {job_id: job}
        if job.get("uses", "").startswith("./.github/workflows/"):
            called = workflow(job["uses"].removeprefix("./.github/workflows/"))["jobs"]
            jobs = {f"{job_id}/{called_id}": called_job for called_id, called_job in called.items()}
        for name, called_job in jobs.items():
            for step in called_job.get("steps", []):
                if "actions/upload-artifact" in step.get("uses", ""):
                    yield name, step["with"].get("name", "artifact")


def test_invariant_check_step_always_runs_and_its_failure_fails_the_release() -> None:
    """No key on the check step, the wheel job, its workflow or a caller can skip the check or absorb its exit code."""
    reusable = workflow("python-wheels.yaml")
    assert set(reusable) == {"name", True, "permissions", "jobs"}
    assert set(reusable["jobs"]) == {"build"}
    job = reusable["jobs"]["build"]
    assert set(job) == {"name", "runs-on", "timeout-minutes", "env", "strategy", "steps"}
    assert job["strategy"] == {"fail-fast": False, "matrix": {"include": job["strategy"]["matrix"]["include"]}}
    check = job["steps"][check_step_index(job["steps"])]
    assert set(check) == {"name", "shell", "env", "run"}
    assert check["shell"] == "bash"
    assert set(check["env"]) == {"WHEEL_PLATFORM", "WHEEL_MACHO_ARCH"}

    assert set(workflow("publish.yaml")["jobs"]["build-python-wheels"]) == {"name", "needs", "if", "uses", "with"}
    assert set(workflow("ci-python-wheels.yaml")["jobs"]["wheels"]) == {"name", "if", "uses"}

    publish = publish_pypi_job()
    assert "build-python-wheels" in publish["needs"]
    assert publish["if"] == PYPI_PUBLISH_CONDITION


def test_invariant_pypi_publishes_exactly_the_files_the_check_step_read() -> None:
    """The upload follows the check and reads its directory; PyPI reads only the wheel and sdist artifacts."""
    steps = wheel_job()["steps"]
    check = check_step_index(steps)
    build = step_index(steps, "xberg-io/actions/build-python-wheels")
    assert build < check
    assert steps[build]["with"]["upload-artifact"] == "false"
    upload = step_index(steps, "actions/upload-artifact")
    assert upload == check + 1
    checked_dir = shlex.split(steps[check]["run"])[2]
    assert posixpath.normpath(steps[upload]["with"]["path"]) == posixpath.join(checked_dir, "*.whl")
    wheel_artifact = steps[upload]["with"]["name"]
    assert wheel_artifact == "python-wheel-${{ matrix.label }}"

    publish = publish_pypi_job()["steps"]
    download = step_index(publish, "actions/download-artifact")
    target = step_index(publish, "xberg-io/actions/publish-pypi")
    pull = publish[download]["with"]
    assert set(pull) == {"pattern", "path", "merge-multiple"}
    assert pull["merge-multiple"] is True
    assert posixpath.normpath(publish[target]["with"]["packages-dir"]) == posixpath.normpath(pull["path"])
    between = publish[download + 1 : target]
    assert [(step["name"], step.get("uses"), step["run"]) for step in between] == [
        ("List packages to publish", None, LIST_PACKAGES_RUN)
    ]

    pattern = pull["pattern"]
    prefix = pattern.removesuffix("*")
    assert prefix and "*" not in prefix and pattern == prefix + "*"
    for label in CONCRETE_TAGS:
        assert fnmatchcase(wheel_artifact.replace("${{ matrix.label }}", label), pattern), label
    reaching = set()
    for job, name in release_run_uploads():
        literal = name.split("${{")[0]
        if literal.startswith(prefix) or ("${{" in name and prefix.startswith(literal)):
            reaching.add((job, name))
    assert reaching == {("build-python-wheels/build", wheel_artifact), ("build-python-sdist", "python-sdist")}


def test_invariant_pypi_checks_every_file_against_the_release_version() -> None:
    """The publish guard compares each file with the PyPI spelling of the version the tag names."""
    publish = publish_pypi_job()
    assert "prepare" in publish["needs"]
    target = publish["steps"][step_index(publish["steps"], "xberg-io/actions/publish-pypi")]
    assert target["with"]["expected-version"] == "${{ needs.prepare.outputs.pypi_version }}"

    prepare = workflow("publish.yaml")["jobs"]["prepare"]
    assert prepare["outputs"]["pypi_version"] == "${{ steps.version_forms.outputs.pypi_version }}"
    assert prepare["outputs"]["checkout_ref"] == "${{ steps.meta.outputs.checkout_ref }}"
    steps = {step.get("id"): step for step in prepare["steps"]}
    assert set(steps["meta"]) == {"name", "id", "uses", "with"}
    forms = steps["version_forms"]
    assert set(forms) == {"name", "id", "shell", "env", "run"}
    assert forms["env"]["VERSION"] == "${{ steps.meta.outputs.version }}"
    assert forms["env"]["RELEASE_TAG"] == "${{ steps.meta.outputs.tag }}"
    assert forms["run"] == "task ci:release-version-forms"


def test_invariant_release_builds_one_wheel_per_tag_and_no_other() -> None:
    """Five legs, five tags, each leg's tag pattern accepts its own tag and no other leg's."""
    legs = wheel_job()["strategy"]["matrix"]["include"]
    labels = [leg["label"] for leg in legs]
    assert sorted(labels) == sorted(CONCRETE_TAGS)
    optional = {"wheel_macho_arch", "before_build_macos"}
    for leg in legs:
        assert {"label", "runner", "cibw_archs_macos", "wheel_platform"} <= set(leg) <= {
            "label",
            "runner",
            "cibw_archs_macos",
            "wheel_platform",
            *optional,
        }
        accepted = {label for label, tag in CONCRETE_TAGS.items() if re.fullmatch(leg["wheel_platform"], tag)}
        assert accepted == {leg["label"]}, leg["label"]
        assert (leg.get("wheel_macho_arch") is not None) == CONCRETE_TAGS[leg["label"]].startswith("macosx_")


def test_invariant_wheel_path_holds_no_permission_beyond_the_upload() -> None:
    """The build has read access only; the PyPI job has the OIDC token and no secret."""
    assert workflow("publish.yaml")["permissions"] == {"contents": "read"}
    reusable = workflow("python-wheels.yaml")
    pull_request = workflow("ci-python-wheels.yaml")
    assert "pull_request" in pull_request[True]
    for checked in (reusable, pull_request):
        assert checked["permissions"] == {"contents": "read"}
        for text in strings(checked):
            for word in ("secrets", "github.token", "GITHUB_TOKEN", "id-token", "twine", "gh-action-pypi-publish"):
                assert word not in text, (word, text)
            assert "maturin publish" not in text and "maturin upload" not in text, text

    publish = publish_pypi_job()
    assert set(publish) == {"name", "needs", "if", "runs-on", "permissions", "timeout-minutes", "environment", "steps"}
    assert publish["permissions"] == {"contents": "read", "id-token": "write"}
    assert publish["environment"]["name"] == "pypi"
    for text in strings(publish):
        assert "secrets" not in text and "github.token" not in text, text
