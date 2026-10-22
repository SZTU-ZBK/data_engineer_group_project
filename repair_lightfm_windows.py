"""Rebuild installed LightFM 1.17 with an unsigned RNG shift on Windows.

Requires Microsoft C++ Build Tools. No dependency installation is performed.
The original extension is backed up; replacement happens only after the rebuilt
extension passes isolated WARP/BPR training and prediction checks.
"""

import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory

from src.models.lightfm import WINDOWS_RNG_CAST


def main():
    if sys.platform != "win32":
        raise SystemExit("This repair is only needed on Windows.")
    spec = importlib.util.find_spec("lightfm")
    if spec is None:
        raise SystemExit("Install lightfm=1.17 first.")
    package = Path(spec.origin).parent
    source = package / "_lightfm_fast_no_openmp.c"
    if not source.is_file():
        raise SystemExit(f"Generated LightFM C source is missing: {source}")
    original = source.read_text(encoding="utf-8")
    original_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
    patched, count = WINDOWS_RNG_CAST.subn(lambda match: f'({match["call"]} >> 1)', original)
    if count == 0:
        print("The unsafe RNG cast was not found; no files were changed.")
        return
    if count != 1:
        raise SystemExit("Unexpected C source layout; refusing to patch.")
    from setuptools import Distribution, Extension
    from setuptools.command.build_ext import build_ext

    with TemporaryDirectory(prefix="lightfm-rng-") as temporary:
        temporary = Path(temporary)
        patched_source = temporary / source.name
        patched_source.write_text(patched, encoding="utf-8")
        extension_name = "lightfm._lightfm_fast_no_openmp"
        distribution = Distribution({"name": "lightfm-rng-repair", "ext_modules": [
            Extension(extension_name, sources=[str(patched_source)]),
        ]})
        command = build_ext(distribution)
        command.build_lib = str(temporary / "lib")
        command.build_temp = str(temporary / "build")
        command.ensure_finalized()
        print("Compiling the corrected extension...", flush=True)
        command.run()
        built = Path(command.get_ext_fullpath(extension_name))
        # Load the candidate extension in a child; the installed binary stays untouched.
        smoke = r'''
import ctypes, importlib.util, sys
ctypes.windll.kernel32.SetErrorMode(0x8002)
name = 'lightfm._lightfm_fast_no_openmp'
spec = importlib.util.spec_from_file_location(name, sys.argv[1])
module = importlib.util.module_from_spec(spec)
sys.modules[name] = module
spec.loader.exec_module(module)
import numpy as np
from scipy.sparse import csr_matrix
from lightfm import LightFM
from src.models.lightfm import LightFMRecommender, build_item_features
b = csr_matrix([[1,1,0,0],[1,0,1,0],[0,0,1,1]], dtype='float32')
genres = csr_matrix([[1,0],[0,1],[1,1],[0,1]], dtype='float32')
for loss in ['warp', 'bpr']:
    for seed in [7, 42]:
        for features in [build_item_features(4), build_item_features(4, genres)]:
            model = LightFM(no_components=4, loss=loss, random_state=seed)
            model.fit(b, item_features=features, epochs=30, num_threads=1)
            exported = LightFMRecommender.from_fitted(model, features, [10,30,80])
            for index, user in enumerate([10,30,80]):
                expected = model.predict(index, np.arange(4), item_features=features, num_threads=1)
                assert np.isfinite(expected).all()
                np.testing.assert_allclose(exported.score_tokens(user, []), expected, rtol=1e-5, atol=1e-5)
print('PASS: WARP/BPR, two seeds, ID/genre features, exported-score parity.')
'''
        print("Checking training and predictions in an isolated process...", flush=True)
        result = subprocess.run([sys.executable, "-c", smoke, str(built)],
                                cwd=Path(__file__).resolve().parent, capture_output=True, text=True,
                                timeout=120, creationflags=subprocess.CREATE_NO_WINDOW)
        if result.returncode:
            raise RuntimeError(f"Rebuilt extension failed; installed files were not replaced.\n{result.stderr}")
        target = package / built.name
        for path in [source, target]:
            backup = path.with_name(path.name + ".before-rng-fix")
            if not backup.exists():
                shutil.copy2(path, backup)
        shutil.copy2(built, target)
        source.write_text(patched, encoding="utf-8")
        metadata = {"fix": "Unsigned temper(seed) >> 1; avoids signed Windows long conversion",
                    "source_sha256_before": original_sha256,
                    "source_sha256_after": hashlib.sha256(source.read_bytes()).hexdigest(),
                    "extension_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                    "smoke_test": result.stdout.strip()}
        (package / "windows_rng_repair.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(result.stdout.strip())
        print("Repair installed. Run: python train.py --model lightfm")


if __name__ == "__main__":
    main()
