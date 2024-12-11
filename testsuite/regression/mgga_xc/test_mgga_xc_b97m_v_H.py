
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_xc_b97m_v_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_xc_b97m_v", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-7.281961649050125e-01, -6.190915903692855e-01, -3.507857220335567e-01, -1.336654538722447e-01, -1.286081586674605e-02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_xc_b97m_v_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_xc_b97m_v", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-1.007403753643762e+00, -9.136603593112818e-01, -8.711837726512813e-01, -4.204551580036499e-01, -4.585020916207570e-01, -5.429382606949797e-02, -1.578788887885117e-01, 8.879296770792361e-02, -1.653539807903406e-02, -1.199433871225562e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_xc_b97m_v_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_xc_b97m_v", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [5.179740229047939e-02, 0.000000000000000e+00, -1.187885737014138e+22, -8.698005591359462e-02, 0.000000000000000e+00, -1.167500781698916e+22, -3.663653981592218e-01, 0.000000000000000e+00, -9.109009847990680e+21, 6.388256910582061e+01, 0.000000000000000e+00, 3.370281668298809e+15, 1.736662757170215e+07, 0.000000000000000e+00, 1.643151863636853e+14]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_xc_b97m_v_H_2_vtau():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_xc_b97m_v", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [6.840181192258672e-02, 2.998530144260043e-04, 4.070122377605263e-02, 3.829089310040810e-01, -7.120146631957835e-03, 9.210149656351828e-01, -1.035199988316217e-02, -5.320911371868132e+00, -2.713589484569056e-06, -5.391108515623592e-01]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
