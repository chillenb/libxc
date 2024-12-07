
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_xc_th_fcfo_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_xc_th_fcfo", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-8.586180249397016e-01, -8.219570760672773e-01, -5.953150269609928e-01, -3.335707832904752e-01, -2.675774613640677e-01]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_xc_th_fcfo_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_xc_th_fcfo", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-1.053962477346523e+00, -9.810055145247676e-02, -9.135858222476719e-01, 5.224370609708062e-02, -5.979610235337319e-01, 9.837909969975484e-02, -2.020909285834570e-01, -1.987021656056213e-02, -8.884942153841333e-02, 1.399513809023485e-01]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_xc_th_fcfo_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_xc_th_fcfo", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [6.439663159536692e-02, -6.427633054410514e-04, 9.152113775204996e+17, -4.457613032551157e-02, -4.636325121878083e-03, 6.665342476904439e+17, -3.343429040870317e-01, -1.480651856804786e-01, -5.748929815899500e+17, -2.433949537500697e+01, -1.613405578204331e+01, -3.264164356348114e+18, -1.953070366406355e+05, 1.309960332684395e+06, -6.241909183487357e+18]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
