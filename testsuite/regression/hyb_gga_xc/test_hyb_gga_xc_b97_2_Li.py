
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_gga_xc_b97_2_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_b97_2", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.508509029558539e+00, -1.070361852337202e+00, -3.646359988451415e-01, -1.359160772921599e-01, -6.823077445006701e-02, -1.459811939310620e-02, -2.975441904930048e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_gga_xc_b97_2_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_b97_2", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-1.966230977279198e+00, -1.967853682610527e+00, -1.350666567285401e+00, -1.351688461005420e+00, -2.554274326427627e-01, -2.560390049474367e-01, -1.826238773538700e-01, 5.540929886507353e-01, -4.429321987821577e-02, 3.576663206118215e-01, -2.063938014986884e-02, -1.976570927651768e-02, -6.425485816107178e-04, 2.861059468847457e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_gga_xc_b97_2_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_b97_2", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-4.988591573952446e-05, 0.000000000000000e+00, -4.959469584396061e-05, -3.309309982224390e-04, 0.000000000000000e+00, -3.295584180178304e-04, -1.134641816165658e-01, 0.000000000000000e+00, -1.131803282537243e-01, 1.039737146520727e+00, 0.000000000000000e+00, 7.697211141215861e+01, -1.012066584595479e+02, 0.000000000000000e+00, 9.224947065572253e+03, -4.532505742018454e-01, 0.000000000000000e+00, -3.031246457364140e-01, -4.857823541413629e+00, 0.000000000000000e+00, 1.519990170073978e+01])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
