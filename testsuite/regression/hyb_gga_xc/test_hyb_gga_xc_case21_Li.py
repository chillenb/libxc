
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_gga_xc_case21_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_case21", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.405272597501974e+00, -1.008115586939650e+00, -3.183594213290265e-01, -1.354693753744905e-01, -6.411969327243389e-02, -1.480792702011819e-02, -2.765144167929175e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_gga_xc_case21_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_case21", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-1.795329649235259e+00, -1.796809798288105e+00, -1.232596289458779e+00, -1.233532950508821e+00, -3.437679931916900e-01, -3.438773095845825e-01, -1.772687093036066e-01, -1.233424340912079e-01, -6.937514698949591e-02, 2.365461046327527e-01, -1.981243153556756e-02, -1.967020636504388e-02, -3.991889576675891e-04, -2.837876649522570e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_gga_xc_case21_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_case21", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-1.007562156213885e-04, 0.000000000000000e+00, -1.003272854529988e-04, -5.077490587832621e-04, 0.000000000000000e+00, -5.059711209102574e-04, -3.776065215717452e-02, 0.000000000000000e+00, -3.763958379755361e-02, -2.022158730798016e-01, 0.000000000000000e+00, 6.854267482363155e+02, -3.359320939544499e+01, 0.000000000000000e+00, 1.927555452952247e+07, -1.262323594398036e-02, 0.000000000000000e+00, -2.539385516726216e-04, 3.215831485686539e-01, 0.000000000000000e+00, 2.274065005963517e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
