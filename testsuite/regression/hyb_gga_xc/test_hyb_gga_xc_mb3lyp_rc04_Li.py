
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_gga_xc_mb3lyp_rc04_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_mb3lyp_rc04", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.520197906236799e+00, -1.102323787981512e+00, -3.454783525876619e-01, -1.350738711774331e-01, -6.677260169693859e-02, -1.002085887263698e-01, -3.867248519136875e-02])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_gga_xc_mb3lyp_rc04_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_mb3lyp_rc04", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-1.891900332865855e+00, -1.893387846517558e+00, -1.317499111114046e+00, -1.318389532837984e+00, -3.863646812853639e-01, -3.866242518277244e-01, -1.728437251506391e-01, -2.861674383765039e-01, -6.366943914931451e-02, -1.213305324409322e+00, -3.203625562511265e-02, -3.225282616952784e-02, -5.446607077523131e-03, -4.793959227682057e-03])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_gga_xc_mb3lyp_rc04_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_mb3lyp_rc04", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-1.919101935266067e-04, 4.230480491699886e-06, -1.913582478680785e-04, -7.091371939437872e-04, 2.954022849291356e-05, -7.073439109924753e-04, -5.453687912656912e-02, 3.866747919504811e-02, -5.438428174813650e-02, -3.175110157836661e+00, 3.722869163256963e+00, -9.617113315518748e+02, -5.514852799530517e+01, 1.909121184807025e+01, -3.492216945794654e+07, -8.386710703408741e+02, 6.428238830639903e-02, -8.400092928335994e+02, -1.036801610330859e+08, 0.000000000000000e+00, -3.088520081156214e+08])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
