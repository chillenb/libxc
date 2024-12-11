
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_x_ncap_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_ncap", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.796468028192982e+00, -1.287207976717874e+00, -4.486321995195552e-01, -1.603137751210418e-01, -8.278310163153049e-02, -6.125527676723949e-01, -1.181489165410009e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_x_ncap_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_ncap", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-2.237941683066776e+00, -2.240081623374749e+00, -1.511545305750184e+00, -1.512918242372725e+00, -3.122102944655541e-01, -3.119581689037565e-01, -2.050969581949041e-01, 3.073645969139697e-01, -7.024448274988328e-02, 3.240837280684102e-01, 3.011994138841999e-01, 3.086647065024339e-01, 3.696838233906202e-01, 3.467298470860268e-01])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_x_ncap_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_ncap", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-2.679048684182504e-04, 0.000000000000000e+00, -2.669739769254649e-04, -1.075058423920316e-03, 0.000000000000000e+00, -1.071580873960908e-03, -1.386404657803219e-01, 0.000000000000000e+00, -1.386685729921301e-01, -4.112122625767518e+00, 0.000000000000000e+00, -1.088554760141378e+04, -8.781987145783729e+01, 0.000000000000000e+00, -1.240161843363132e+09, -9.221341047389435e+03, 0.000000000000000e+00, -9.345631603074187e+03, -4.263928288392473e+09, 0.000000000000000e+00, -1.366889884785422e+10])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
