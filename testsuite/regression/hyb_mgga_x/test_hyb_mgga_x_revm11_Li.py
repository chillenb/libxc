
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_mgga_x_revm11_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_revm11", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.042132544429555e+00, -6.499894494249630e-01, -1.899319750220592e-01, -2.492382869972625e-02, -3.631933510493349e-03, -7.742627382029859e-05, -3.879802602739144e-10])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_mgga_x_revm11_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_revm11", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-1.793144700237193e+00, -1.795827632783954e+00, -7.914795468758428e-01, -7.940886156242778e-01, -2.336271526938611e-01, -2.339471488854606e-01, -5.094369073072327e-02, -1.338672605072652e-04, -5.119201127378278e-03, -4.319475156690112e-09, -1.558255310330356e-04, -1.521731973684905e-04, -1.287229147011463e-09, -2.304502391352476e-25])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_x_revm11_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_revm11", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([1.909485789779521e-04, 0.000000000000000e+00, 1.899516284471652e-04, 4.770300639688127e-04, 0.000000000000000e+00, 4.780316993249290e-04, -1.832441392436283e-02, 0.000000000000000e+00, -1.829676191057073e-02, 1.276457986258049e-01, 0.000000000000000e+00, -4.934411638190359e-03, -3.712806014120737e+00, 0.000000000000000e+00, -3.185076033280944e-05, -5.550645643803810e-03, 0.000000000000000e+00, -5.101638197174405e-03, -1.034449481120727e-05, 0.000000000000000e+00, 0.000000000000000e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_x_revm11_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_revm11", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([3.504445754808150e-02, 3.519990977580958e-02, -3.041555672494656e-02, -3.020132139650264e-02, -6.846406574913443e-03, -6.771288430092601e-03, 1.606671870641910e-01, -2.555801867216061e-08, -2.530989099201682e-03, -5.273310801237437e-15, -1.399665663425910e-11, -3.005726953252904e-08, -3.201772850247455e-26, 0.000000000000000e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
