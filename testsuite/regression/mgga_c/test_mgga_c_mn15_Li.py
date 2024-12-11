
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_c_mn15_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_mn15", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-2.780962951655567e-01, -1.317336949810430e-01, -2.568092444892244e-01, -5.308874279428695e-02, -4.816962479246942e-02, -1.922820721630476e-03, -4.775266027879906e-05]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_c_mn15_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_mn15", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-6.710405017689317e-01, -6.705588085550547e-01, -3.593356902244748e-01, -3.591174196244323e-01, -6.139803858366511e-02, -6.162343229411971e-02, -1.320003076802353e-01, -4.352038623638434e-01, -7.621122092597228e-03, -2.644118397930329e-01, -2.413976350807923e-03, -2.441097368718543e-03, -5.617609055536476e-05, -8.243054258470189e-05]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_mn15_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_mn15", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [4.857719334182911e-04, 9.715438668365822e-04, 4.857719334182911e-04, 6.972315071458766e-04, 1.394463014291753e-03, 6.972315071458766e-04, 1.256003670993285e+00, 2.512007341986571e+00, 1.256003670993285e+00, 1.421683708825985e+01, 2.843367417651971e+01, 1.421683708825985e+01, 1.068470278094987e+03, 2.136940556189975e+03, 1.068470278094987e+03, 1.132320360398267e-07, 2.264640782099350e-07, 1.132320360398267e-07, 1.192712712146426e-15, -2.109789916791679e-14, 1.192712712146426e-15]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_mn15_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_mn15", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [5.198823653612548e-02, 5.198823653612546e-02, 3.125434717469387e-02, 3.125434717469387e-02, -4.265911580040415e-02, -4.265911580040412e-02, 1.908896487609748e+00, 1.908896487609332e+00, -4.184746170398269e-01, -4.184746167513680e-01, -2.569193587466248e-08, -2.569193587466249e-08, -6.789917492745246e-20, -6.790563720735420e-20]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
