
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_c_wi0_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_wi0", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-5.164346623774568e-02, -4.685540312193548e-02, 4.875898714721858e-04, -3.404445429015793e-02, -7.393166450511054e-05, -1.552990692053925e-09, -8.722642844107701e-17])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_c_wi0_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_wi0", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-6.099808246265189e-02, -6.099808246265189e-02, -6.984281658271246e-02, -6.984281658271246e-02, -5.638096867317951e-03, -5.638096867317951e-03, -5.382719319795302e-02, -5.382719319795302e-02, -2.289550766215263e-03, -2.289550766215263e-03, -9.317942909063192e-09, -9.317942909063192e-09, -5.233585706462791e-16, -5.233585706462791e-16])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_c_wi0_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_wi0", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([7.393317689267028e-06, 1.478663537853406e-05, 7.393317689267028e-06, 5.689026833373957e-05, 1.137805366674791e-04, 5.689026833373957e-05, 1.521808129918372e-03, 3.043616259836744e-03, 1.521808129918372e-03, 7.522657230524016e+00, 1.504531446104803e+01, 7.522657230524016e+00, 4.795122023296511e+00, 9.590244046593023e+00, 4.795122023296511e+00, 2.999137539857158e-05, 5.998275079714315e-05, 2.999137539857158e-05, 6.826556292103777e-07, 1.365311258420755e-06, 6.826556292103777e-07])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
