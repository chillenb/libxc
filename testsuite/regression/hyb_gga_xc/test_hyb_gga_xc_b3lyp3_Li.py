
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_gga_xc_b3lyp3_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_b3lyp3", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.499383333374459e+00, -1.081921910294254e+00, -3.323293523522787e-01, -1.316183571903835e-01, -6.557132222936282e-02, -9.979407358434814e-02, -3.867766254971960e-02])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_gga_xc_b3lyp3_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_b3lyp3", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-1.870965271146755e+00, -1.872491423303709e+00, -1.296417164493497e+00, -1.297342980823601e+00, -3.708167124095397e-01, -3.710651931472738e-01, -1.683948652080080e-01, -1.178943481347759e-01, -6.193696663003702e-02, -4.408758264187203e-02, -3.141789984854553e-02, -3.163077141306647e-02, -5.461217611585088e-03, -4.779087514063004e-03])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_gga_xc_b3lyp3_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_b3lyp3", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-1.919101935266067e-04, 4.230480491699886e-06, -1.913582478680785e-04, -7.091371939437872e-04, 2.954022849291356e-05, -7.073439109924753e-04, -5.453687912656912e-02, 3.866747919504811e-02, -5.438428174813650e-02, -3.175110157836661e+00, 3.722869163256963e+00, -9.617113315518748e+02, -5.514852799530517e+01, 1.909121184807025e+01, -3.492216945794654e+07, -8.386710703408741e+02, 6.428238830639903e-02, -8.400092928335994e+02, -1.036801610330859e+08, 0.000000000000000e+00, -3.088520081156214e+08])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
