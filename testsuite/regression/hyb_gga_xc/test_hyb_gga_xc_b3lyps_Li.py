
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_gga_xc_b3lyps_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_b3lyps", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.590278467745653e+00, -1.146180877495074e+00, -3.502688117063091e-01, -1.426478268875790e-01, -7.123144473968074e-02, -1.017319235249592e-01, -3.876193972289004e-02])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_gga_xc_b3lyps_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_b3lyps", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-1.990830880991853e+00, -1.992453400146383e+00, -1.380828876762848e+00, -1.381816920645754e+00, -3.937352076669291e-01, -3.939766242050181e-01, -1.822538357726433e-01, -1.181056407626381e-01, -6.887795937147570e-02, -4.501281342903429e-02, -3.377709281163708e-02, -3.397771489128074e-02, -5.574761302789205e-03, -4.876350109739531e-03])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_gga_xc_b3lyps_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_b3lyps", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-1.919101935266067e-04, 4.230480491699886e-06, -1.913582478680785e-04, -7.091371939437872e-04, 2.954022849291356e-05, -7.073439109924753e-04, -5.453687912656912e-02, 3.866747919504811e-02, -5.438428174813650e-02, -3.175110157836661e+00, 3.722869163256963e+00, -9.617113315518748e+02, -5.514852799530517e+01, 1.909121184807025e+01, -3.492216945794654e+07, -8.386710703408741e+02, 6.428238830639903e-02, -8.400092928335994e+02, -1.036801610330859e+08, 0.000000000000000e+00, -3.088520081156214e+08])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
