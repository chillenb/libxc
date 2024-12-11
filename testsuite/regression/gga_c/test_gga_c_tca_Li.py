
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_c_tca_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_tca", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-6.124373773131012e-02, -4.871961162324535e-02, -4.762806912203593e-03, -1.103288216786306e-02, -9.054895202739977e-04, -2.500348751622994e-07, -1.034914048976975e-12])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_c_tca_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_tca", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-9.191742907613364e-02, -9.180711418970432e-02, -9.365254918538073e-02, -9.356964973350351e-02, -1.850337528733350e-02, -1.850742712377444e-02, -1.816720626446984e-02, -3.132263233159767e-01, -3.528630660174814e-03, -3.283826850852404e-01, -1.095172070201662e-06, -1.098822538020336e-06, -4.352498955663310e-12, -5.112505606751076e-12])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_c_tca_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_tca", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([2.372102317526549e-05, 4.744204635053099e-05, 2.372102317526549e-05, 1.098466498679218e-04, 2.196932997358437e-04, 1.098466498679218e-04, 3.137688002693389e-03, 6.275376005386779e-03, 3.137688002693389e-03, 2.247977158135287e+00, 4.495954316270574e+00, 2.247977158135287e+00, 5.175065247319054e+00, 1.035013049463811e+01, 5.175065247319054e+00, 3.172880463779583e-03, 6.345760927559166e-03, 3.172880463779583e-03, 5.322524281555739e-03, 1.064504856311148e-02, 5.322524281555739e-03])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
