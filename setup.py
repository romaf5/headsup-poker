"""Build the optional C++ extension:  python setup.py build_ext --inplace

The extension (`headsup_cpp`) speeds up DeepCFR traversals ~50x and provides an
envpool-style vectorised environment.  Everything works without it (pure Python fallbacks).
"""

import os
import sys

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

if sys.platform == "darwin" and os.path.exists("/usr/bin/clang++"):
    # Homebrew's LLVM (if first in PATH) ships a libc++ whose headers don't resolve against
    # the Xcode SDK; Apple's clang always works for Python extensions.
    os.environ.setdefault("CC", "/usr/bin/clang")
    os.environ.setdefault("CXX", "/usr/bin/clang++")


class BuildExt(build_ext):
    """pyenv-built Pythons on macOS bake `-I<SDK>/usr/include` into CFLAGS, which shadows
    libc++'s wrapper headers and breaks every C++ compile.  Strip those flags."""

    def build_extensions(self):
        if sys.platform == "darwin":
            bad = lambda f: f.startswith("-I") and "/SDKs/" in f and f.endswith("/usr/include")
            for attr, flags in list(vars(self.compiler).items()):
                if attr.startswith(("compiler", "linker")) and isinstance(flags, list):
                    setattr(self.compiler, attr, [f for f in flags if not bad(f)])
        super().build_extensions()


extra = ["-O3"]
if sys.platform == "darwin":
    extra += ["-mmacosx-version-min=11.0"]

setup(
    name="headsup_cpp",
    version="0.1.0",
    ext_modules=[
        Pybind11Extension(
            "headsup_cpp",
            ["headsup/cpp/headsup_cpp.cpp"],
            cxx_std=17,
            extra_compile_args=extra,
        )
    ],
    cmdclass={"build_ext": BuildExt},
    zip_safe=False,
)
