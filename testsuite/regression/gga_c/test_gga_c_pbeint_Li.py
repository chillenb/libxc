
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_c_pbeint_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_pbeint", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-6.481249212961580e-02, -4.864024140212887e-02, -4.496995857151145e-03, -1.575579924670539e-02, -1.897411582425819e-03, -1.252548281673732e-08, -2.994295349861842e-16]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_c_pbeint_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_pbeint", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-1.171029124557365e-01, -1.169703772482292e-01, -1.044139408936294e-01, -1.043118635045203e-01, -2.203561892140888e-02, -2.204359192600467e-02, -2.369220970060800e-02, -1.036637268181247e-01, -8.613443069188747e-03, 4.282123203130603e-01, -8.105347741150699e-08, -8.146065781374963e-08, -1.873914706133212e-15, -2.219477043558654e-15]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_c_pbeint_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_pbeint", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [4.225450228858988e-05, 8.450900457717979e-05, 4.225450228858988e-05, 1.418840289584825e-04, 2.837680579169650e-04, 1.418840289584825e-04, 4.260021414533830e-03, 8.520042829067664e-03, 4.260021414533830e-03, 2.761184344476328e+00, 5.522368688952657e+00, 2.761184344476328e+00, 1.442803884096014e+01, 2.885607768192028e+01, 1.442803884096014e+01, 2.762497724129753e-04, 5.524995448211986e-04, 2.762497724129753e-04, 2.645004959677917e-06, 5.290252715233887e-06, 2.645004959677917e-06]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
