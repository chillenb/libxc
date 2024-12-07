
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_gdme_0_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_gdme_0", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [3.208061028711579e+01, -9.036463161125088e-01, -9.105138932324806e-01, -1.014110532688417e-01, -8.978408144611480e-02, -2.944845841389886e+01, 1.851128668722434e+02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_gdme_0_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_gdme_0", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-1.584109842546140e+01, -1.583424759595846e+01, -3.269107956094238e+00, -3.271345951308178e+00, -5.542545642335981e-01, -5.535911043323478e-01, -4.348226597224149e-01, 1.079183934371422e+01, -1.544208660572905e-01, -1.516548054274434e+01, 9.762883996206030e+00, 9.802326592579780e+00, -5.459449220467298e+01, -8.149530817594663e+01]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_gdme_0_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_gdme_0", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_gdme_0_Li_2_vlapl():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_gdme_0", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vlapl"].flatten()
    ref_tgt = [-2.128663886333459e-02, -2.126747629382624e-02, -3.065945887755229e-02, -3.063338549419230e-02, -1.276243353153691e-01, -1.276786334294296e-01, -2.353620224729970e-01, -3.396636789385828e+00, -5.937131331895656e-01, -1.070974376325344e+02, -3.230068637911486e+00, -3.253733301140237e+00, -1.603392940513488e+02, -2.255413506798399e+02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_gdme_0_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_gdme_0", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [8.514655545333838e-02, 8.506990517530497e-02, 1.226378355102092e-01, 1.225335419767692e-01, 5.104973412614765e-01, 5.107145337177182e-01, 9.414480898919881e-01, 1.358654715754331e+01, 2.374852532758263e+00, 4.283897505301376e+02, 1.292027455164595e+01, 1.301493320456095e+01, 6.413571762053950e+02, 9.021654027193597e+02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
