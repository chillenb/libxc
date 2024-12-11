
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_xc_vv10_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_xc_vv10", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.861210091082818e+00, -1.351125729584346e+00, -4.311054702891443e-01, -1.750202931970261e-01, -8.492089596276872e-02, -4.918352671011425e-02, -3.814340998249822e-03])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_xc_vv10_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_xc_vv10", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-2.299179876052598e+00, -2.301266043953376e+00, -1.556818647090282e+00, -1.558032924696090e+00, -4.336683337552005e-01, -4.336896616215282e-01, -2.272718869583426e-01, -1.361213806077959e-01, -8.907508201254644e-02, 3.388269061017621e-01, -3.963881154604996e-02, -3.971678669510767e-02, -3.234305074551539e-03, -2.542679096334946e-03])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_xc_vv10_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_xc_vv10", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-3.267140410283948e-04, 9.190971700708733e-05, -3.252942252792839e-04, -1.365767717770856e-03, 2.980993506782570e-04, -1.361139057961880e-03, -7.105830466620795e-02, 6.249948659585063e-03, -7.094868586541592e-02, -1.553098758558659e+00, 6.762268918356340e+00, -2.382820150020991e+02, -5.202799256351922e+01, 2.258698854598489e+01, -2.023695250110584e+06, -2.153751888945098e+02, 3.357174600576258e-04, -2.133508210188484e+02, -4.694867223785895e+06, 3.212885779437900e-06, -1.229067417938484e+07])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
