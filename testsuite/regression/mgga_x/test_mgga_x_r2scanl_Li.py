
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_r2scanl_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_r2scanl", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.872327685557759e+00, -1.364193392521450e+00, -3.286291578446273e-01, -1.648943118745998e-01, -7.184056519569061e-02, -9.507971292321905e-03, -1.821094706455403e-05]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08

# test_mgga_x_r2scanl_Li_2_vrho() not generated due to NaN in reference data

# test_mgga_x_r2scanl_Li_2_vsigma() not generated due to NaN in reference data

# test_mgga_x_r2scanl_Li_2_vlapl() not generated due to NaN in reference data
