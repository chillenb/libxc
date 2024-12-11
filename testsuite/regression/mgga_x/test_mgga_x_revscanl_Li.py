
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_revscanl_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_revscanl", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.871323669661477e+00, -1.369901998248573e+00, -3.286291578446273e-01, -1.635195422059603e-01, -7.184056519569061e-02, -9.263559814292565e-03, -7.304891950207407e-05]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08

# test_mgga_x_revscanl_Li_2_vrho() not generated due to NaN in reference data

# test_mgga_x_revscanl_Li_2_vsigma() not generated due to NaN in reference data

# test_mgga_x_revscanl_Li_2_vlapl() not generated due to NaN in reference data
