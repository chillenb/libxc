
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_c_rscan_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_rscan", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-7.424616477180734e-16, -9.610368056911511e-16, -4.019354293838262e-15, -8.633718739936569e-14, -1.955045359543911e-11])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_c_rscan_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_rscan", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-7.632783294297951e-17, -2.198763077912139e-01, -1.561251128379126e-16, -2.061881205506179e-01, -1.415534356397075e-15, -1.731868428931188e-01, -6.860831347488272e-14, -7.071811968129105e-02, -1.895111931965454e-11, -1.674245713528410e-03])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_rscan_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_rscan", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([2.896102147959951e-17, 5.792204295919903e-17, 2.896102147959951e-17, 5.517609352401333e-17, 1.118434328189459e-16, 5.517609352401333e-17, 2.520804250192276e-15, 5.041608500384552e-15, 2.520804250192276e-15, 6.397735602428926e-12, 1.279547120485785e-11, 6.397735602428926e-12, 1.044514593613507e-05, 2.089029187227014e-05, 1.044514593613507e-05])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_rscan_H_2_vtau():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_rscan", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([-1.820245022231922e-21, -1.798905354343978e-21, -5.922933159054752e-19, -5.826382676070761e-19, -4.866122583553658e-18, -4.848441818732104e-18, -2.761328545051615e-22, -2.761257011808913e-22, -3.370901254858265e-27, -3.370901281139636e-27])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
