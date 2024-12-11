
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_c_vsxc_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_vsxc", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-7.266304328285467e-02, -4.679608288602948e-02, -8.448662068097615e-01, -2.923984944547002e-02, -6.237265979229468e-02, 8.359561434618888e-07, -3.373280763332118e-08]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_c_vsxc_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_vsxc", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-1.253425085447231e-01, -1.255291806062966e-01, -9.056357270854827e-02, -9.074865821001710e-02, 9.821683188389891e-01, 9.624093634787131e-01, -4.852115496815428e-02, 5.196717263272237e-03, 1.688532610826800e-01, 7.946904499151726e-06, 3.085125626888968e-06, -1.164080361766669e-04, -5.226387688402111e-15, -3.810403214705405e-07]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_vsxc_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_vsxc", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [1.822356618798212e-04, 0.000000000000000e+00, 1.829498351311189e-04, 1.693173749589385e-04, 0.000000000000000e+00, 1.718510370312576e-04, 6.530836388712586e+00, 0.000000000000000e+00, 6.457368161826720e+00, 3.734115792232165e+01, 0.000000000000000e+00, -3.867356862662478e+01, 1.952528931515352e+03, 0.000000000000000e+00, -4.868442132522893e+03, 3.033132453205971e-04, 0.000000000000000e+00, 2.578313245369698e+00, 9.469666884817439e-10, 0.000000000000000e+00, 3.571024917637090e+04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_vsxc_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_vsxc", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [6.464690451272410e-03, 6.493921538529039e-03, 5.681747790222633e-03, 5.705462551314264e-03, -3.560238477936857e-01, -3.570106251010904e-01, 4.119264657102072e-01, 3.904624238217849e-05, -2.038295404750724e+00, 2.638258718169071e-07, -1.542193855175163e-08, -3.747288159335614e-05, -1.540481646926016e-19, 1.076429708450036e-07]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
