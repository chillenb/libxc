
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_2d_prhg07_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_2d_prhg07", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-4.257324985182684e+00, -2.341934366852386e+00, -4.531281390805099e-01, -1.157150410226883e-01, -4.981818941377072e-02, -1.088818744015364e-01, -2.571785465372694e-02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_2d_prhg07_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_2d_prhg07", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-6.381668137951860e+00, -6.390295163079676e+00, -3.731699105025410e+00, -3.735843734935415e+00, -4.239996507781247e-01, -4.238145727016028e-01, -1.735762388076427e-01, -4.736066498058100e-02, -3.981197062995682e-02, -2.964817678603798e-03, -4.733514534659878e-02, -4.711444848537056e-02, -1.132529343910863e-03, -5.244258187443114e-05]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_2d_prhg07_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_2d_prhg07", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, -1.609645932536218e-03, 0.000000000000000e+00, -1.598698531147234e-03, -9.494559306753587e-02, 0.000000000000000e+00, -9.490947484192862e-02, 0.000000000000000e+00, 0.000000000000000e+00, -6.464365769316066e+02, -7.394431732929823e+01, 0.000000000000000e+00, -2.771025595612665e+07, -5.701854697416913e+02, 0.000000000000000e+00, -5.758658116607609e+02, -8.573682880539595e+07, 0.000000000000000e+00, -2.604136400804419e+08]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_2d_prhg07_Li_2_vlapl():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_2d_prhg07", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vlapl"].flatten()
    ref_tgt = [0.000000000000000e+00, 0.000000000000000e+00, -6.988322285950551e-03, -6.958531747957228e-03, -5.714922660009006e-03, -5.705463335447598e-03, 0.000000000000000e+00, -2.064023415524982e-03, -4.420901736309005e-02, -2.822534905390954e-03, -2.116980795382434e-03, -2.091758125645280e-03, -2.602456866287078e-03, -2.840018852542711e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_2d_prhg07_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_2d_prhg07", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [0.000000000000000e+00, 0.000000000000000e+00, 2.795328914380220e-02, 2.783412699182891e-02, 2.285969064003602e-02, 2.282185334179039e-02, 0.000000000000000e+00, 8.256093662099926e-03, 1.768360694523602e-01, 1.129013962156382e-02, 8.467923181529736e-03, 8.367032502581121e-03, 1.040982746514831e-02, 1.136007541017085e-02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
