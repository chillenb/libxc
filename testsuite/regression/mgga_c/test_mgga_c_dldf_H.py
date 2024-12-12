
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_c_dldf_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_dldf", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-4.856654648977450e-11, -5.492772681650548e-12, -2.328562243010614e-12, -3.431665605847890e-13, 1.729005683918689e-09])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_c_dldf_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_dldf", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-3.156022010971134e-02, -2.557890552031126e-01, -1.026887713441949e-02, -2.884385816856577e-01, -6.434955449188540e-03, -2.800585985522183e-01, -7.510243263672289e-03, -5.711780051789969e-02, -9.845411080381468e-04, 2.959418859262870e-02])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_dldf_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_dldf", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([1.290913757617048e+00, 0.000000000000000e+00, -3.230690123786396e+21, 1.271943981326560e-02, 0.000000000000000e+00, -2.583790159206648e+21, 3.747604081434919e-02, 0.000000000000000e+00, -1.305938753353080e+21, 2.254273686143791e+00, 0.000000000000000e+00, 2.325468324691238e+20, 2.097935476557420e+03, 0.000000000000000e+00, 3.196366532722626e+14])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_dldf_H_2_vtau():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_dldf", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([-3.078809432433191e+00, 0.000000000000000e+00, -2.186839698213441e-02, 0.000000000000000e+00, -1.289254512673691e-02, 0.000000000000000e+00, -1.477084780299174e-02, 0.000000000000000e+00, -1.437290195633615e-03, 0.000000000000000e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
