
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_lda_xc_gdsmfb_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_xc_gdsmfb", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-6.525553850364357e-01, -5.872578805210316e-01, -3.506858670099325e-01, -1.003735741491912e-01, -5.690483705494816e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_lda_xc_gdsmfb_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_xc_gdsmfb", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-8.629576914485929e-01, -3.230709257564134e-01, -7.762723838000026e-01, -3.080652978497835e-01, -4.625592669099318e-01, -2.367622060704565e-01, -1.318062546930188e-01, -1.010963445452407e-01, -7.494495438482758e-03, -7.189745238454185e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
