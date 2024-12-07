
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_gga_xc_camy_b3lyp_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_camy_b3lyp", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.447964126579745e+00, -1.026130951646818e+00, -2.588221442415757e-01, -9.080011219089505e-02, -3.562740989433884e-02, -4.964219531045006e-02, -1.882018408119269e-02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_gga_xc_camy_b3lyp_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_camy_b3lyp", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-1.820571650119619e+00, -1.822110426352926e+00, -1.243396804649632e+00, -1.244324472721535e+00, -3.356721874216762e-01, -3.359488090778553e-01, -1.239346926231333e-01, -1.037214526979308e-01, -3.676581753877056e-02, -4.137907476743857e-02, -1.672593130903466e-02, -1.687729302165578e-02, -2.664649189129744e-03, -2.388227293082789e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_gga_xc_camy_b3lyp_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_camy_b3lyp", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-1.999813323135684e-04, 4.230480491699886e-06, -1.994164864892619e-04, -7.146884705198476e-04, 2.954022849291356e-05, -7.129440185106701e-04, -3.346438596132571e-02, 3.866747919504811e-02, -3.330351806730645e-02, -2.081571424403915e+00, 3.722869163256963e+00, -4.660638271179674e+02, -2.818046861365577e+01, 1.909121184807025e+01, -1.697604723954102e+07, -4.076711492185145e+02, 6.428238830639903e-02, -4.083215331616164e+02, -5.040007827997237e+07, 0.000000000000000e+00, -1.501363928339827e+08]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
