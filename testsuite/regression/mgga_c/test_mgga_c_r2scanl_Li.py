
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_c_r2scanl_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_r2scanl", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-4.965801390648251e-02, -3.185372393017430e-02, -1.449166625055379e-02, -1.226253420987959e-02, 2.802618331854134e-09, -5.846194648235521e-04, -5.302292720117550e-07])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08

# test_mgga_c_r2scanl_Li_2_vrho() not generated due to NaN in reference data

# test_mgga_c_r2scanl_Li_2_vsigma() not generated due to NaN in reference data

# test_mgga_c_r2scanl_Li_2_vlapl() not generated due to NaN in reference data
