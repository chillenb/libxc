
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_c_m11_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_m11", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-3.942963112639320e-02, -5.104142966733265e-02, -1.418660291624656e-02, -1.791480837725306e-02, -1.333950984959590e-02])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_c_m11_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_m11", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([9.195276081950744e-02, -1.116971019122557e+03, -8.415688360111891e-02, -2.093372531628257e+03, -3.960202246800602e-02, 2.100156204316402e+02, 3.468447139299129e-02, 2.385067877674947e+01, -1.694570231336260e-02, -6.002380609035603e-02])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_m11_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_m11", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-9.869326636863113e-01, -1.973865327372622e+00, -9.869326636863113e-01, -2.878055007382657e-02, -5.756110014765314e-02, -2.878055007382657e-02, 2.127761599646715e-02, 4.255523199293430e-02, 2.127761599646715e-02, 6.154904835147347e-01, 1.230980967029475e+00, 6.154904835147347e-01, 1.124538192176073e-02, 2.249076382209516e-02, 1.124538192176073e-02])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_m11_H_2_vtau():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_m11", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([-4.502141592891703e+00, -4.499872848234352e+00, 1.086897449971172e-01, 1.079614321264887e-01, 1.699181280883923e-02, 1.695766702781170e-02, -7.218422060549015e-02, -7.218378822388724e-02, -5.581331867287988e-05, -5.581331899308733e-05])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
