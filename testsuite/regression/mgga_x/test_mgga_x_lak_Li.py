
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_lak_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_lak", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.994254931794011e+00, -1.325657311570214e+00, -2.394683436094193e-01, -1.821720739185917e-01, -5.194706938828260e-02, -4.816589060100795e-03, -3.700486833039759e-05])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_lak_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_lak", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-2.763978778181596e+00, -2.766466863188256e+00, -1.995433480498962e+00, -1.996835487128289e+00, -3.209516349386182e-01, -3.209431685726882e-01, -2.473618871865537e-01, -1.124692927353207e-02, -7.159318922857126e-02, -1.105765742303149e-04, -5.911150697106171e-03, -1.184713626818029e-02, -3.384378308926088e-06, -1.803805359287682e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_lak_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_lak", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-2.857911904187373e-04, 0.000000000000000e+00, -2.845862349347644e-04, -1.852148044683455e-03, 0.000000000000000e+00, -1.844609012685585e-03, 1.030988884549781e-01, 0.000000000000000e+00, 1.031450723662639e-01, -3.206831069474494e+00, 0.000000000000000e+00, 2.805042349174488e+01, 3.964741906927966e+01, 0.000000000000000e+00, 2.662357800677746e+04, 3.064038837207293e-01, 0.000000000000000e+00, 2.519790213334071e+01, 1.940994653209789e-02, 0.000000000000000e+00, 1.021769468959947e+12])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_lak_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_lak", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([1.513619557724577e-02, 1.511255876992466e-02, 3.345899762439477e-02, 3.340390167359609e-02, 2.922587718020460e-04, 3.174333815865877e-04, 1.248288085556496e-01, 8.990639119150012e-13, 1.951251945989458e-02, 7.239760800208528e-14, 1.957664304543306e-14, 9.141107136030131e-13, 0.000000000000000e+00, 2.489067319268457e-19])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
