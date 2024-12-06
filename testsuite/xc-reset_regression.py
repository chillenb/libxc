#!/usr/bin/env python3
# MPL License Info
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

import os
import sys
from subprocess import run, DEVNULL
from multiprocessing import Pool
from dataclasses import dataclass, field


# Reference density systems
systems = ["H", "Li", "BrOH", "BrOH+"]

# Maximum differentiation order to test
MAX_ORDER = 2  # set to 3 to get FXC


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

    if not ctx.xc_reg:
        ctx.xc_reg = os.path.join(ctx.builddir, "xc-regression")

    if not os.path.exists(ctx.xc_reg):
        print("xc-regression not found")
        return 1

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

        for system in systems:
            for pol, nspin in (("pol", 2), ("unpol", 1)):
                for order in range(MAX_ORDER):
                    refname = f"{func}.{system}.{pol}.{order}"

                    yield (
                        ctx,
                        func,
                        nspin,
                        order,
                        system,
                        dir,
                        refname,
                    )


def gen_one_ref(args):
    "Create a single reference file."
    ctx, func, nspin, order, system, dir, refname = args

    os.makedirs(os.path.join(ctx.destdir, dir), exist_ok=True)
    dest = os.path.join(ctx.destdir, dir, refname)

    # Real work is delegated to a C program.
    p = run(
        [
            ctx.xc_reg,
            func,
            str(nspin),
            str(order),
            os.path.join(ctx.srcdir, "input", system),
            dest,
        ],
        stdout=DEVNULL,
    )

    if p.returncode != 0:
        print("failed to generate", refname)
        return (refname, p.returncode)

    # If the generation succeed we replace the old archive.
    if os.path.exists(dest + ".bz2"):
        os.remove(dest + ".bz2")
    run(["bzip2", dest], stdout=DEVNULL)

    return (refname, 0)


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(
        description="Regenerate the reference files for the testsuite"
    )

    parser.add_argument(
        "--srcdir", "-s", default=None, help="Specifiy the source directory (default: ./)"
    )
    parser.add_argument(
        "--builddir", "-b", default=None, help="Specifiy the build directory (default: ./)"
    )
    parser.add_argument(
        "--destdir", "-d", default=None, help="Specifiy a destination directory (default: <srcdir>/regression)"
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
