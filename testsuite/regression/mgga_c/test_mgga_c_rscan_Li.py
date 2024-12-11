
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_c_rscan_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_rscan", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-4.075571356256243e-02, -5.616362088841530e-02, -6.497298454587860e-02, -1.716676347939049e-03, -9.564266217519414e-04, -2.952822060372091e-04, -1.198664416700895e-07]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_c_rscan_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_rscan", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-1.290398923618959e-02, -1.275588740345585e-02, 1.415851740697699e-03, 1.627991625651656e-03, -6.818896086389614e-02, -6.824634522412977e-02, 7.973420101315722e-04, -1.591846832924629e-01, -1.070404716638898e-03, -1.073113624052790e-01, -5.636691370007008e-04, -5.704605770989033e-04, -2.017734140081034e-07, -3.435000721302030e-07]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_rscan_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_rscan", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [7.194024593630309e-05, 1.438804918726062e-04, 7.194024593630309e-05, 5.725766132304157e-04, 1.145153226460831e-03, 5.725766132304157e-04, 2.483625902813614e-01, 4.967251805627227e-01, 2.483625902813614e-01, 2.750626596561033e+00, 5.501253193122067e+00, 2.750626596561033e+00, 1.761858675981226e+01, 3.523717351962450e+01, 1.761858675981226e+01, 3.266845972808068e-02, 6.533691945616135e-02, 3.266845972808068e-02, 9.364091937379513e-04, 1.872818387475903e-03, 9.364091937379513e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_rscan_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_rscan", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [-4.302296036278418e-03, -4.302296036278417e-03, -9.148059316286790e-03, -9.148059316286787e-03, -1.283950533025587e-03, -1.283950533025586e-03, -9.334878652694902e-02, -9.334878652692841e-02, -1.659808881610107e-02, -1.659808880269137e-02, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
