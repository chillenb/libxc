
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_vt84_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_vt84", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.750891674124534e+00, -1.209547767302443e+00, -2.978069275945001e-01, -1.585817051688221e-01, -6.346075275880425e-02, -5.550059698020617e-03, -4.481603741388305e-05]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_vt84_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_vt84", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.349654527773770e+00, -2.351787123923825e+00, -1.628229006377466e+00, -1.629649543151194e+00, -3.936466244987612e-01, -3.932943742571211e-01, -2.124563825657449e-01, -1.528597631044643e-02, -8.273689464235236e-02, -1.189923457218070e-09, -5.420334483280774e-03, -1.590218259885505e-02, -2.395527953715692e-15, -2.260694856305692e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_vt84_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_vt84", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-3.416211653146179e-04, 0.000000000000000e+00, -3.386765669594526e-04, -9.571721176095924e-04, 0.000000000000000e+00, -9.551550725498638e-04, -2.460213182891439e-02, 0.000000000000000e+00, -2.538838473134306e-02, -7.431210567024905e+00, 0.000000000000000e+00, 1.697306963912622e+01, -2.779427944923961e+01, 0.000000000000000e+00, 6.033158796239914e-01, 7.900991452005776e-01, 0.000000000000000e+00, 1.427925674679184e+01, 2.753508334006695e-11, 0.000000000000000e+00, 2.570411508879611e+10]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_vt84_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_vt84", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [2.333558921446658e-03, 2.314714424977653e-03, 2.368553206842056e-03, 2.372788744696770e-03, -6.446925943312984e-04, -6.822206698443516e-04, 2.697740212272849e-02, -5.692649747610254e-09, -1.571988716964751e-02, -1.624128224460650e-17, -6.957686337564865e-12, -5.787831298205824e-09, -6.658038599908378e-34, -5.590213714322106e-13]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
