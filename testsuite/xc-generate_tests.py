#!/usr/bin/env python3
# MPL License Info
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

import os
import sys
import math
import pylibxc
from pylibxc.example_densities import test_data
from subprocess import run, DEVNULL
from multiprocessing import Pool
from dataclasses import dataclass, field

thresholds = {'zk' : 5e-8, 'vrho' : 5e-5, 'vsigma' : 5e-5, 'vtau' : 5e-5, 'vlapl' : 5e-5}

@dataclass
class Context:
    srcdir: str
    builddir: str
    destdir: str
    funcs: list[str] = field(default_factory=list)
    xc_reg: str = ""

if sys.stdout.isatty():
    NC = "\033[0m"
    RED = "\033[1;31m"
    GREEN = "\033[1;32m"
    YELLOW = "\033[1;33m"
else:
    NC = RED = GREEN = YELLOW = ""

def main(ctx: Context, nproc=os.cpu_count()):
    "Main driver. Run reference generation in parallel."

    if ctx.funcs:
        ctx.funcs = ctx.funcs
    else:
        # Collect the list of functionals from the header
        with open(os.path.join(ctx.builddir, "../src/xc_funcs.h"), "r") as f:
            ctx.funcs = [
                line.split()[1].lower()[len("xc_") :]
                for line in f
                if line.startswith("#define")
            ]

    failed = []
    with Pool(processes=nproc) as p:
        for i, (r, rcode) in enumerate(p.imap(gen_one_ref, enum_refs(ctx), chunksize=50)):
            if rcode != 0:
                print(f"{RED}F{NC}", end="")
                failed.append(r)
            else:
                print(f"{GREEN}.{NC}", end="")

            if (i + 1) % 80 == 0:
                print()

        if (i + 1) % 80 != 0:
            print()

    if failed:
        print(f"{RED}{len(failed)} failure(s){NC}")
        for f in failed:
            print(f)
        return 1
    else:
        print(f"{GREEN}Done.{NC}")
        return 0


def enum_refs(ctx: Context):
    "Generate the test set from the list of functionals."
    for func in ctx.funcs:
        if func.startswith("hyb"):
            _, fn_kind, component, *rest = func.split("_")
            dir = f"hyb_{fn_kind}_{component}"
        else:
            fn_kind, component, *rest = func.split("_")
            dir = f"{fn_kind}_{component}"

        for system in test_data:
            refname = f"test_{func}_{system}.py"
            yield (
                ctx,
                func,
                system,
                dir,
                refname,
            )


def gen_one_ref(args):
    "Create a single reference file."
    ctx, func, system, dir, refname = args

    # Input data for the system
    inp = test_data[system]
    nspin = len(inp["rho"].shape)

    # Get the functional
    feval = pylibxc.LibXCFunctional(func, nspin)

    # We only test first and second derivatives
    do_l = False
    do_k = False
    do_f = False
    do_v = feval._have_vxc
    do_e = feval._have_exc

    # Evaluate the data
    out = feval.compute(inp, do_exc=do_e, do_vxc=do_v, do_fxc=do_f, do_kxc=do_k, do_lxc=do_l)

    test_targets = []
    # Check if functional has energy
    if "zk" in out:
        test_targets.append('zk')
    # Add all first derivatives
    for target in out:
        if target.startswith('v'):
            test_targets.append(target)

    os.makedirs(os.path.join(ctx.destdir, dir), exist_ok=True)
    dest = os.path.join(ctx.destdir, dir, refname)

    # Write the test file header
    fout = open(dest, 'w')
    fout.write(f'''
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data
''')

    # Write the tests
    for target in test_targets:
        fout.write(f'''

def test_{func}_{system}_{nspin}_{target}():
    # Prepare the input
    inp = test_data["{system}"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("{func}", {nspin})

    # Evaluate the data
    out = feval.compute(inp, do_exc={do_e}, do_vxc={do_v}, do_fxc={do_f}, do_kxc={do_k}, do_lxc={do_l})
    tgt = out["{target}"].flatten()
    ref_tgt = [''')
        # Print out reference value we just computed
        for ival, val in enumerate(out[target].flatten()):
            if ival:
                fout.write(', ')
            # Special handling for NaN
            if math.isnan(val):
                fout.write(f'"nan"')
            else:
                fout.write(f'{val:.15e}')
        fout.write(f''']
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < {thresholds[target]}
''')
    fout.close()

    return (refname, 0)


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(
        description="Regenerate the reference files for the testsuite"
    )

    parser.add_argument(
        "--srcdir", "-s", default=None, help="Specify the source directory (default: ./)"
    )
    parser.add_argument(
        "--builddir", "-b", default=None, help="Specify the build directory (default: ./)"
    )
    parser.add_argument(
        "--destdir", "-d", default=None, help="Specify a destination directory (default: <srcdir>/regression)"
    )

    parser.add_argument(
        "--nproc",
        "-j",
        type=int,
        default=os.cpu_count(),
        help=f"Number of concurrent workers (default: {os.cpu_count()})",
    )

    parser.add_argument(
        "funcs",
        metavar="FUNCTIONALS",
        nargs="*",
        default=[],
        help="Restrict generation to a subset of functionals",
    )

    args = parser.parse_args()
    src = args.srcdir or os.getenv("srcdir", "./")
    ctx = Context(
        srcdir=src,
        builddir=args.builddir or os.getenv("builddir", "./"),
        destdir=args.destdir or os.getenv("destdir", os.path.join(src, "regression")),
        funcs=args.funcs,
    )

    sys.exit(main(ctx, nproc=args.nproc))
