
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_xc_pbelyp1w_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_xc_pbelyp1w", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.859820324412842e+00, -1.342122053870566e+00, -4.185125709303502e-01, -1.647295191006197e-01, -8.335624462877815e-02, -2.384905130264262e-02, -4.484473413241769e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_xc_pbelyp1w_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_xc_pbelyp1w", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-2.316710013423109e+00, -2.318677357835173e+00, -1.591006394742601e+00, -1.592214727637558e+00, -4.927698175458397e-01, -4.932717795217916e-01, -2.107794256901593e-01, -1.204904258872967e-01, -7.980929934223516e-02, -4.259116523067712e-02, -3.166793331499621e-02, -3.156260854685125e-02, -6.193147838629062e-04, -5.368726579178734e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_xc_pbelyp1w_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_xc_pbelyp1w", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-2.502659855827141e-04, 3.864883412170266e-06, -2.494789609324350e-04, -9.804811841459629e-04, 2.698736924043955e-05, -9.777231260112154e-04, -4.905977096713713e-02, 3.532584519053778e-02, -4.875826274362851e-02, -3.950452798783427e+00, 3.401139729395250e+00, 2.274208677467011e+00, -6.769668422485137e+01, 1.744135403403949e+01, 1.130456428510798e+01, -2.519758188445281e-01, 5.872712018115467e-02, -2.331442293320584e-01, -1.293192057875573e+00, 0.000000000000000e+00, -1.851071381863682e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
