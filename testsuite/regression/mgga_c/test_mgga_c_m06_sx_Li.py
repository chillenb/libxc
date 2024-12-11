
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_c_m06_sx_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_m06_sx", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-8.937083903044821e-02, -8.315282886817477e-02, -1.422800403231489e-01, -8.472026014762008e-03, -2.989636414436519e-02, 1.471465681374013e-02, 2.375939242875677e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_c_m06_sx_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_m06_sx", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-9.326689684670325e-02, -9.298572264748334e-02, -8.054532146538068e-02, -8.027678695128454e-02, -6.220700740922125e-03, -7.113233536054947e-03, -4.923527951189204e-03, 5.312654625124931e-01, -1.035246298056083e-02, 3.213786077230789e-01, 2.199969817388147e-02, 2.253094616773277e-02, 3.055466941972838e-04, 6.680085896769630e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_m06_sx_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_m06_sx", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([6.232224907472876e-05, 0.000000000000000e+00, 6.153522268469455e-05, 1.522156225081697e-04, 0.000000000000000e+00, 1.525143861621724e-04, 2.044761590809300e-01, 0.000000000000000e+00, 2.095125445393226e-01, 8.536552494508845e+00, 0.000000000000000e+00, 8.828276073056250e+00, 1.475006554290107e+02, 0.000000000000000e+00, -3.318671339342112e+05, -1.560312739913219e+00, 0.000000000000000e+00, -7.512647517695171e+01, -5.310088360500191e+00, 0.000000000000000e+00, -1.995856291754103e+13])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_m06_sx_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_m06_sx", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([-6.948794765085297e-04, -6.964495502692665e-04, -1.666008814211938e-03, -1.674946869317984e-03, -2.845072055460510e-02, -2.892875739496908e-02, -1.454532277966061e-01, 1.040920809263247e-03, -2.126502512751101e-01, 1.395565842941693e-04, 2.320004230993288e-05, 1.091577433834717e-03, 6.447299770256740e-10, -1.204944314354105e-10])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
