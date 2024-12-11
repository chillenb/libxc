
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_lda_xc_1d_ehwlrg_3_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_xc_1d_ehwlrg_3", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-3.416699555976943e+02, -1.570053223748483e+01, -1.303814236610868e-01, -2.950598827462098e-02, -5.450784025277340e-03, -3.724960796155142e-04, -2.335253608265071e-07])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_lda_xc_1d_ehwlrg_3_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_xc_1d_ehwlrg_3", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-1.275106801220851e+03, -1.275106801220851e+03, -6.132111954507199e+01, -6.132111954507199e+01, -2.019841982237199e-01, -2.019841982237199e-01, -4.735946839852344e-02, -4.735946839852344e-02, -8.774090624819420e-03, -8.774090624819420e-03, -5.997172846264977e-04, -5.997172846264977e-04, -3.759758309257337e-07, -3.759758309257337e-07])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
