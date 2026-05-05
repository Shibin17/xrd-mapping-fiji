r"""Run Fiji (ImageJ) headlessly to convert an XRD plot image into a CSV.

Fiji install: https://imagej.net/software/fiji/downloads#installation
On Windows the executable is typically:  <Fiji.app>\ImageJ-win64.exe
Set the env var FIJI_PATH or pass `fiji_path=` explicitly.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path


# A small ImageJ macro: opens an image, takes a horizontal line profile across
# the full width at the vertical centre, and saves the profile as CSV.
# Override by passing a custom macro string to `extract_csv_from_image`.
DEFAULT_MACRO = r"""
args = getArgument();
parts = split(args, "|");
inPath  = parts[0];
outPath = parts[1];
open(inPath);
run("8-bit");
w = getWidth();
h = getHeight();
makeLine(0, h/2, w-1, h/2);
profile = getProfile();
f = File.open(outPath);
print(f, "2theta_deg,Intensity_counts");
for (i = 0; i < profile.length; i++) {
    print(f, i + "," + profile[i]);
}
File.close(f);
run("Quit");
"""


def find_fiji(fiji_path: str | None = None) -> str | None:
    """Locate the Fiji executable. Order: arg → env → PATH → common installs."""
    if fiji_path and Path(fiji_path).exists():
        return fiji_path
    env = os.environ.get("FIJI_PATH")
    if env and Path(env).exists():
        return env
    for name in ("fiji-windows-x64.exe", "ImageJ-win64.exe", "ImageJ-win32.exe", "fiji", "ImageJ-linux64", "Contents/MacOS/ImageJ-macosx"):
        found = shutil.which(name)
        if found:
            return found
    candidates = [
        os.path.expanduser(r"~\Downloads\fiji-latest-win64-jdk\Fiji\fiji-windows-x64.exe"),
        r"C:\Fiji.app\ImageJ-win64.exe",
        r"C:\Program Files\Fiji.app\ImageJ-win64.exe",
        os.path.expanduser(r"~\Fiji.app\ImageJ-win64.exe"),
        os.path.expanduser(r"~\Desktop\Fiji.app\ImageJ-win64.exe"),
        "/Applications/Fiji.app/Contents/MacOS/ImageJ-macosx",
        "/opt/Fiji.app/ImageJ-linux64",
    ]
    for c in candidates:
        if Path(c).exists():
            return c
    return None


def launch_fiji_with_macro(
    macro_path: str | Path,
    fiji_path: str | None = None,
    output_dir: str | Path | None = None,
) -> subprocess.Popen:
    """Launch Fiji (with GUI) and run the given .ijm macro file.

    If `output_dir` is given, the macro is patched in a temp copy so that the
    'Select OUTPUT folder for CSVs' getDirectory() call is replaced with the
    given path — Fiji won't prompt the user to pick the output folder.
    """
    macro_path = Path(macro_path).resolve()
    if not macro_path.exists():
        raise FileNotFoundError(macro_path)
    fiji = find_fiji(fiji_path)
    if fiji is None:
        raise RuntimeError(
            "Fiji executable not found. Install Fiji "
            "(https://imagej.net/software/fiji/downloads#installation) "
            "and either add it to PATH, set FIJI_PATH, or pass fiji_path=..."
        )

    macro_to_run = macro_path
    if output_dir is not None:
        out = Path(output_dir).resolve()
        out.mkdir(parents=True, exist_ok=True)
        # Forward slashes + trailing slash for ImageJ
        out_ij = out.as_posix().rstrip("/") + "/"
        text = macro_path.read_text(encoding="utf-8", errors="replace")
        # Replace the output-folder picker with a literal string
        patched = text.replace(
            'getDirectory("Select OUTPUT folder for CSVs")',
            f'"{out_ij}"',
        )
        tmp = tempfile.NamedTemporaryFile("w", suffix=".ijm", delete=False, encoding="utf-8")
        tmp.write(patched)
        tmp.close()
        macro_to_run = Path(tmp.name)

    cmd = [fiji, "-macro", str(macro_to_run)]
    return subprocess.Popen(cmd)


def extract_csv_from_image(
    image_path: str | Path,
    out_csv: str | Path | None = None,
    fiji_path: str | None = None,
    macro: str = DEFAULT_MACRO,
    timeout: int = 120,
) -> Path:
    """Run Fiji headlessly on `image_path` and return path to the CSV produced."""
    image_path = Path(image_path).resolve()
    if not image_path.exists():
        raise FileNotFoundError(image_path)

    fiji = find_fiji(fiji_path)
    if fiji is None:
        raise RuntimeError(
            "Fiji executable not found. Install Fiji "
            "(https://imagej.net/software/fiji/downloads#installation) "
            "and either add it to PATH, set FIJI_PATH, or pass fiji_path=..."
        )

    if out_csv is None:
        out_csv = Path(tempfile.gettempdir()) / (image_path.stem + "_fiji.csv")
    out_csv = Path(out_csv).resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile("w", suffix=".ijm", delete=False) as mf:
        mf.write(macro)
        macro_file = mf.name

    # Forward slashes are friendlier to ImageJ macro string parsing
    arg = f"{image_path.as_posix()}|{out_csv.as_posix()}"
    cmd = [fiji, "--headless", "--console", "-macro", macro_file, arg]

    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
        )
    finally:
        try:
            os.unlink(macro_file)
        except OSError:
            pass

    if not out_csv.exists():
        raise RuntimeError(
            f"Fiji ran but produced no CSV.\nCMD: {' '.join(cmd)}\n"
            f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )
    return out_csv
