
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data

# test_mgga_xc_vcml_rvv10_H_2_zk() not generated due to NaN in reference data

# test_mgga_xc_vcml_rvv10_H_2_vrho() not generated due to NaN in reference data

# test_mgga_xc_vcml_rvv10_H_2_vsigma() not generated due to NaN in reference data


def test_mgga_xc_vcml_rvv10_H_2_vtau():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_xc_vcml_rvv10", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([1.176822765380585e-20, 0.000000000000000e+00, -9.381605470549845e-03, 0.000000000000000e+00, 4.633115599016871e-03, 0.000000000000000e+00, 1.437275762755948e-04, 0.000000000000000e+00, 1.364146744331377e-11, 0.000000000000000e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
