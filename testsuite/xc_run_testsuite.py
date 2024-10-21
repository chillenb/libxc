#!/usr/bin/env python3
# MPL License Info
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

import os
import sys
import shutil
from subprocess import run, PIPE, DEVNULL
import tempfile
import glob
from multiprocessing import Pool
from dataclasses import dataclass

# Check if SKIP_CHECK is set
if os.getenv("SKIP_CHECK"):
    print("Skipping checks")
    sys.exit(0)


@dataclass
class Context:
    srcdir: str
    builddir: str
    workdir: str
    selfunc: str


# Color definitions for terminal output
if sys.stdout.isatty():
    NC = "\033[0m"
    RED = "\033[1;31m"
    GREEN = "\033[1;32m"
    YELLOW = "\033[1;33m"
else:
    NC = RED = GREEN = YELLOW = ""


def main(ctx, nproc=int(1.5 * os.cpu_count())):
    failed = []
    with Pool(nproc) as p:
        for i, (res, name, workfiles) in enumerate(
            p.imap_unordered(run_one_test, enum_tests(ctx), chunksize=50)
        ):
            if res == "1":
                print(f"{GREEN}.{NC}", end="")
                for f in workfiles:
                    os.remove(f)
            else:
                print(f"{RED}F{NC}", end="")
                failed.append((name, res))

            if (i + 1) % 80 == 0:
                print()

    if (i + 1) % 80 != 0:
        print()

    return failed


def enum_tests(ctx):
    for full_dir in map(os.path.realpath, glob.glob(f"{ctx.srcdir}/regression/*/")):
        for reffile in glob.glob(full_dir + "/" + ctx.selfunc):
            refname = os.path.splitext(os.path.basename(reffile))[0]

            yield (ctx, full_dir, refname)


def run_one_test(t):
    ctx, full_dir, refname = t

    func, system, pol, order = refname.split(".")
    nspin = 1 if pol == "unpol" else 2

    if order == "0":
        label = "exc"
        tol = 5e-8
    elif order == "1":
        label = "vxc"
        tol = 5e-5
    elif order == "2":
        label = "fxc"
        tol = 5e-4

    ref_file = os.path.join(ctx.workdir, refname + "_ref")
    res_file = os.path.join(ctx.workdir, refname)
    name = f"{func:<30} {system:<11} {nspin:<7} {label:<11}"

    # Extract the reference
    with open(ref_file, "wb") as out_file:
        run(["bunzip2", "-c", os.path.join(full_dir, refname + ".bz2")], stdout=out_file)

    # Evaluate the functional
    p = run(
        [
            os.path.join(ctx.builddir, "xc-regression"),
            func,
            str(nspin),
            order,
            os.path.join(ctx.srcdir, "input", system),
            res_file,
        ],
        stdout=DEVNULL,
    )
    if p.returncode != 0:
        return ("-1", name, [ref_file])

    # Compare to the reference
    p = run(
        [os.path.join(ctx.builddir, "xc-error"), res_file, ref_file, str(tol)],
        text=True,
        stdout=PIPE,
    )
    res = p.stdout.strip()

    return (res, name, [ref_file, res_file])


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Run the testsuite of libxc")

    parser.add_argument(
        "--workdir", "-w", default=None, help="Specifiy the working directory"
    )
    parser.add_argument(
        "--srcdir", "-s", default=None, help="Specifiy the source directory"
    )
    parser.add_argument(
        "--builddir", "-b", default=None, help="Specifiy the build directory"
    )

    parser.add_argument(
        "--nproc",
        "-j",
        type=int,
        # Oversuscribing a bit gives a fair speedup
        default=int(1.5 * os.cpu_count()),
        help="Number of concurrent workers",
    )

    parser.add_argument(
        "func",
        metavar="FUNCTIONAL",
        nargs="?",
        default=None,
        help="Restrict testing to a single functional",
    )

    opts = parser.parse_args()

    ctx = Context(
        srcdir=opts.srcdir or os.getenv("srcdir", "./"),
        builddir=opts.builddir or os.getenv("builddir", "./"),
        workdir=opts.workdir
        or os.getenv("workdir", tempfile.mkdtemp(prefix="/tmp/libxc.")),
        selfunc=f"{opts.func}.*.bz2" if opts.func else "*.bz2",
    )

    print(f"{YELLOW}Comparing against reference data{NC}")
    print(f"Using {ctx.workdir} as working directory")

    failed = main(ctx, nproc=opts.nproc)

    if failed:
        print("FAILED:")
        print(
            "========================================================================="
        )
        print("  Functional                     System      NSpin   Quantity    Result")
        print(
            "========================================================================="
        )
        for n, result in failed:
            print(" ", n, result)
    else:
        shutil.rmtree(ctx.workdir)

    sys.exit(len(failed))
