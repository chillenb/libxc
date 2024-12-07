
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_c_hyb_tau_hcth_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_hyb_tau_hcth", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-8.006836321881308e-02, -7.117238660793022e-02, 1.330200899050856e-02, -1.469818134587231e-02, -6.182533055326352e-03, 1.426188091650230e-02, 2.301594012447062e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_c_hyb_tau_hcth_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_hyb_tau_hcth", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-9.421443145787015e-02, -9.415619995585003e-02, -7.834769379255162e-02, -7.826571343529524e-02, -1.201992618551602e-01, -1.208403252359196e-01, -2.435554090534737e-02, 5.871901998924398e-01, -7.365194174156015e-03, 3.556453424691938e-01, 1.792030447557192e-02, 1.851802843711179e-02, 1.218537442194225e-04, 8.224024500168826e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_c_hyb_tau_hcth_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_hyb_tau_hcth", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [1.082633723343248e-05, 0.000000000000000e+00, 1.094453568545145e-05, -1.644198244900390e-06, 0.000000000000000e+00, -1.299003381975688e-06, 6.563505264688253e-02, 0.000000000000000e+00, 6.585925829208859e-02, 3.431608636390067e+00, 0.000000000000000e+00, 6.705781903599990e+01, -6.093215364675372e-02, 0.000000000000000e+00, 7.890974044647434e+03, 8.135597260491939e-01, 0.000000000000000e+00, 8.625639300281757e-01, 1.357425193225011e+00, 0.000000000000000e+00, 2.087459805384088e+01]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
