
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_regtpss_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_regtpss", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.827690445041067e+00, -1.217854400271803e+00, -2.909439081905955e-01, -1.701258155160065e-01, -6.259926156462302e-02, -2.054733655625132e-02, -3.490521354631697e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_regtpss_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_regtpss", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-2.636619758182198e+00, -2.639415797047580e+00, -1.702282593589242e+00, -1.704306158204008e+00, -3.893433720954116e-01, -3.891809464496629e-01, -2.416952738914786e-01, -2.609111414186120e-02, -8.325806562889272e-02, -8.296413305158533e-04, -2.750646504909760e-02, -2.723266425828722e-02, -5.541564195078384e-04, -2.183758224394548e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_regtpss_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_regtpss", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-6.077613403798828e-04, 0.000000000000000e+00, -6.048250456573293e-04, -1.348174113955801e-03, 0.000000000000000e+00, -1.348006010105945e-03, -1.232232224483908e-01, 0.000000000000000e+00, -1.235110440526793e-01, -1.045257511869361e+01, 0.000000000000000e+00, -4.352781648750259e-01, -6.167207559181653e+01, 0.000000000000000e+00, -2.785411081860444e+00, -1.855375773948758e-04, 0.000000000000000e+00, -4.130531457693814e-01, -1.271365602272227e-10, 0.000000000000000e+00, -1.260262022228873e+12])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_regtpss_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_regtpss", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([2.863250984471429e-02, 2.864423640521448e-02, 1.158531993269837e-02, 1.166523861011116e-02, 2.450185069324343e-04, 2.468997695535103e-04, 4.208829476143497e-01, 1.181165623758641e-10, -1.738484008866524e-03, 6.061104628947541e-17, 1.320718701593170e-15, 1.354661166472653e-10, 2.484942007651405e-33, 1.003194952912577e-12])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
