
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_c_tau_hcth_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_tau_hcth", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-5.926841276283806e-02, -7.183807296963320e-02, 1.269012084639545e-02, -2.776623423359934e-03, -1.147381093652308e-02, 2.978600699168261e-02, 4.807311556994247e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_c_tau_hcth_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_tau_hcth", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [2.283845135741096e-03, 2.284668414089969e-03, -2.804499727942623e-03, -2.736335053506537e-03, -2.198160793275328e-01, -2.209845619535433e-01, 2.458606159543771e-03, 1.221928670079901e+00, -9.663047025510381e-03, 7.439987762524982e-01, 3.736150599856880e-02, 3.860622885550610e-02, 2.538346950894380e-04, 1.719679550588644e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_c_tau_hcth_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_tau_hcth", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-1.154103395311067e-04, 0.000000000000000e+00, -1.147775486504524e-04, -4.035362099064011e-04, 0.000000000000000e+00, -4.016639443527566e-04, 1.135932927200750e-01, 0.000000000000000e+00, 1.140004450806189e-01, -2.963619958778412e+00, 0.000000000000000e+00, 1.854742667184942e+02, -8.878290085205846e+00, 0.000000000000000e+00, 2.187513668460570e+04, 2.244095578795404e+00, 0.000000000000000e+00, 2.380329581380296e+00, 3.698243780085649e+00, 0.000000000000000e+00, 5.777424300440315e+01]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
