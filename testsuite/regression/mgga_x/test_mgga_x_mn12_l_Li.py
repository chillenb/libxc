
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_mn12_l_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_mn12_l", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.381197429194172e+00, -1.055025794930873e+00, -1.737472585735031e-02, -1.079149783859832e-01, -4.539308072810359e-02, -5.074129773702428e-02, -7.835529233615601e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_mn12_l_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_mn12_l", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-1.290188147758042e+00, -1.286723613410144e+00, -1.487534320654376e+00, -1.488688869831364e+00, -3.503827420550917e-01, -3.585525033074561e-01, -2.133830700742768e-01, -6.331134159067489e-02, -1.047349790339846e-01, -2.156595436974718e-03, -6.712686459665172e-02, -6.586253285943518e-02, -1.441446316611859e-03, 6.037523759398771e-05]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_mn12_l_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_mn12_l", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-5.280044177624372e-03, 0.000000000000000e+00, -5.272720579800888e-03, -9.397359515479728e-03, 0.000000000000000e+00, -9.401999336970394e-03, -1.202244663682014e+00, 0.000000000000000e+00, -1.173728092785213e+00, 4.434125984256320e+01, 0.000000000000000e+00, -3.985522051607108e+00, -5.321119766619278e+02, 0.000000000000000e+00, -2.716518481681883e+01, -1.703062293791328e-03, 0.000000000000000e+00, -3.770934312068575e+00, -1.240668084148745e-09, 0.000000000000000e+00, -2.591715670517347e+13]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_mn12_l_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_mn12_l", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [-6.631176485576649e-02, -6.721736656569502e-02, 2.646180818842329e-02, 2.648922154009161e-02, 6.978271121017683e-02, 7.133131486308149e-02, 1.984664631340136e+00, -1.387664712027105e-05, 3.828361803333054e-01, -2.907986186411262e-09, -6.906741493565611e-09, -1.496002199027692e-05, -3.955766297069638e-20, 8.703330940425597e-11]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
