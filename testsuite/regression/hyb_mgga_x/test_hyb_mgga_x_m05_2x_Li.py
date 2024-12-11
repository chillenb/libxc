
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_mgga_x_m05_2x_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_m05_2x", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-5.967238689732053e-01, -4.532572417311803e-01, -1.820932564114643e-01, -5.884854512341181e-02, -3.314375410327272e-02, -3.398199022480825e-02, -5.780592581122719e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_mgga_x_m05_2x_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_m05_2x", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-6.949774852640714e-01, -6.974214313408935e-01, -5.398079105359627e-01, -5.407650973019179e-01, -1.833880069057019e-01, -1.849319462677918e-01, -1.205365278812048e-01, -4.297287857980580e-02, -2.391128286844937e-02, -1.373933976683875e-03, -4.554724636213672e-02, -4.483786864369479e-02, -9.177278065052683e-04, -3.616508733975752e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_x_m05_2x_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_m05_2x", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-9.550504963313425e-05, 0.000000000000000e+00, -9.515116290200166e-05, -4.497331826607264e-04, 0.000000000000000e+00, -4.481105136140700e-04, -2.264640912893305e-01, 0.000000000000000e+00, -2.257856042654081e-01, -1.557433190533934e+00, 0.000000000000000e+00, -4.586118502742421e-01, -8.958015827053528e+01, 0.000000000000000e+00, -2.941927703927742e+00, -1.959534664185842e-04, 0.000000000000000e+00, -4.351295303387527e-01, -1.342814019475314e-10, 0.000000000000000e+00, -5.799750993973732e+12]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_x_m05_2x_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_m05_2x", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [-1.426636984199866e-02, -1.400737893018895e-02, -9.406179161223722e-03, -9.330120070293725e-03, -1.090455659779754e-02, -1.066327035567555e-02, 1.196584200994538e+00, -3.018231609631271e-05, -1.698071285956461e-01, -6.168896743427449e-09, -1.497757453961943e-08, -3.258072019469985e-05, -8.399674815073785e-20, -3.815052333394285e-10]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
