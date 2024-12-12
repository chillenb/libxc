
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_k_csk1_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_k_csk1", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([4.664152133689035e+00, 6.913102024732256e+00, 3.502998309080055e+00, 6.716629584636133e-02, 7.880461260205794e-02, 5.285880799345448e+01, 8.785715181472222e+04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_k_csk1_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_k_csk1", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-2.738951859324953e+00, -2.733607105043119e+00, 4.911049743327872e+00, 4.928091333195421e+00, -1.224079010347670e-01, -6.999277097179137e-01, 1.164056238119733e-01, -3.386363269636171e-01, -3.568166703064683e-02, -1.210227402333293e+00, -3.357579634641856e-01, -3.475308790934789e-01, -1.576047662274691e-01, -1.185586587645303e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_k_csk1_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_k_csk1", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([1.924481523775052e-02, 0.000000000000000e+00, 1.919279308011938e-02, 4.406812604414295e-02, 0.000000000000000e+00, 4.394024517363030e-02, 1.072000550673467e+00, 0.000000000000000e+00, 1.749691960993734e+00, 2.039828907585682e+01, 0.000000000000000e+00, 8.699790635302614e+03, 3.286709219402713e+02, 0.000000000000000e+00, 2.454376727172780e+09, 7.481638617199318e+03, 0.000000000000000e+00, 7.647285960080892e+03, 9.151270127775303e+08, 0.000000000000000e+00, 2.292358375079780e+10])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_k_csk1_Li_2_vlapl():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_k_csk1", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vlapl"].flatten()
    ref_tgt = numpy.asarray([2.638259693593401e-04, 2.647573102888335e-04, 4.400775997871519e-02, 4.405851380201278e-02, 1.391059668438628e-01, 1.086132238240428e-01, 4.068113270599803e-02, 1.666666666666298e-01, 4.012337574137929e-02, 2.831310028208998e-13, 1.666666666666667e-01, 1.666666666666668e-01, 1.666666666666665e-01, 8.419278807845893e-15])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
