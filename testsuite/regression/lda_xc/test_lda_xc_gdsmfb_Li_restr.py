
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_lda_xc_gdsmfb_Li_restr_1_zk():
    # Prepare the input
    inp = test_data["Li_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_xc_gdsmfb", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.824101645229884e+00, -1.285155906711773e+00, -3.388008256233817e-01, -1.581184456035857e-01, -6.905057441538742e-02, -1.820594847698590e-02, -3.768433465983364e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_lda_xc_gdsmfb_Li_restr_1_vrho():
    # Prepare the input
    inp = test_data["Li_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_xc_gdsmfb", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.410368174275600e+00, -1.694318736981073e+00, -4.419778141062653e-01, -2.054010914599207e-01, -8.969133407460191e-02, -2.381784894637689e-02, -5.007868439528750e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
