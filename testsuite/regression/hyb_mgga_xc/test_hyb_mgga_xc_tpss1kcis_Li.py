
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_mgga_xc_tpss1kcis_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_tpss1kcis", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.754179851265707e+00, -1.223806671620935e+00, -3.586533458982329e-01, -1.546116266962858e-01, -6.668551215640380e-02, -1.791723697959249e-02, -3.339854697409885e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_mgga_xc_tpss1kcis_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_tpss1kcis", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.178661604513869e+00, -2.180371723265829e+00, -1.557948383285905e+00, -1.559302079983530e+00, -3.349367386272709e-01, -3.353694732165591e-01, -2.016533730323851e-01, -1.306682394621697e-01, -6.903616856306616e-02, -4.532973269272330e-02, -2.406019804885203e-02, -2.386921506011776e-02, -4.822124818713236e-04, -3.428680337553541e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_xc_tpss1kcis_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_tpss1kcis", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-7.457631690096728e-04, 6.147937556578381e-05, -7.451113701203192e-04, -1.109938437461127e-03, 2.026042608264738e-04, -1.107674692285387e-03, -7.183637365285873e-02, 1.042811294544413e-02, -7.124351052453615e-02, -5.402827351540765e+00, 4.862987484677894e+00, 2.190154523582114e+00, -3.113497004749520e+01, 2.591897797340521e+01, 1.142051765392939e+01, 2.751900971349948e-01, 1.042064888636250e+00, 2.919840096908375e-01, 1.310501960762925e+02, 2.643505997762938e+02, 1.305716639930497e+02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_xc_tpss1kcis_Li_2_vlapl():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_tpss1kcis", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vlapl"].flatten()
    ref_tgt = [0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_xc_tpss1kcis_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_tpss1kcis", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [3.820316233532314e-02, 3.830346820424212e-02, 1.860833235767865e-02, 1.866480516399061e-02, -5.906442671164532e-04, -7.179176779059414e-04, 1.902083357958112e-01, -2.443752375539020e-06, -4.730308639787827e-02, -1.572871019115246e-08, -1.148406592644299e-09, -2.527401141795257e-06, -3.203183121790221e-19, -3.695772064113588e-09]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
