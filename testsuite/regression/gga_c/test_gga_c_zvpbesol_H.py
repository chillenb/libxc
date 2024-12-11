
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_c_zvpbesol_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_zvpbesol", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-3.218966723947013e-02, -2.649026805025272e-02, -2.179792163387165e-02, -1.322156430798734e-02, -1.575746294245629e-03])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_c_zvpbesol_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_zvpbesol", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-3.681413052380909e-02, 1.157734340303454e-02, -2.610515417101948e-02, 5.465079372345289e+00, -1.313715841128129e-02, 1.904107178923350e+00, -1.472940928763374e-02, -8.587408712843930e-02, -2.004338848365761e-03, -6.542002833382654e-03])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_c_zvpbesol_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_zvpbesol", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([1.136217915354013e-02, 2.272435830708026e-02, 1.136217915354013e-02, -2.579861540840638e-03, -5.159723081681277e-03, -2.579861540840638e-03, -3.218134745715916e-02, -6.436269491431834e-02, -3.218134745715916e-02, -1.252987020882924e-01, -2.505974041765849e-01, -1.252987020882924e-01, -1.279013024018456e-73, -2.558026048036911e-73, -1.279013024018456e-73])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
