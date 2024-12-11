
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_gga_xc_camh_b3lyp_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_camh_b3lyp", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.464045225765634e+00, -1.041360493073447e+00, -2.712970024489673e-01, -9.725291621306251e-02, -4.396318063813389e-02, -6.958543153324218e-02, -2.686228266408265e-02])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_gga_xc_camh_b3lyp_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_camh_b3lyp", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-1.835913673519691e+00, -1.837456308818370e+00, -1.257047571687862e+00, -1.257978990014321e+00, -3.407631508991054e-01, -3.410308852639434e-01, -1.299764162168446e-01, -1.090292514617879e-01, -4.156622772559825e-02, -4.253771077746920e-02, -2.214985429561798e-02, -2.232925881979698e-02, -3.783834092883557e-03, -3.356191790677743e-03])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_gga_xc_camh_b3lyp_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_camh_b3lyp", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-2.033668035787690e-04, 4.230480491699886e-06, -2.027885353787140e-04, -7.310913587670099e-04, 2.954022849291356e-05, -7.292865821787186e-04, -3.780406038010888e-02, 3.866747919504811e-02, -3.764498515997354e-02, -2.341859374563280e+00, 3.722869163256963e+00, -6.670015930285241e+02, -3.837658885392027e+01, 1.909121184807025e+01, -2.425150219294857e+07, -5.824003702807075e+02, 6.428238830639903e-02, -5.833296382168040e+02, -7.200011182853191e+07, 0.000000000000000e+00, -2.144805611914038e+08])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
