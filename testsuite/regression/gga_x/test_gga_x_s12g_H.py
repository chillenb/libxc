
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_x_s12g_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_s12g", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-6.455061028159940e-01, -5.819735873578101e-01, -3.595372231919294e-01, -1.361426906895740e-01, -7.204686321799242e-03])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_x_s12g_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_s12g", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-8.606727428940181e-01, -8.711890851991050e-17, -7.552726503116826e-01, -1.585746421527254e-16, -3.850449787150627e-01, -8.954730007619259e-17, -1.480511194068020e-01, -7.592909289975314e-17, -9.601567913396312e-03, -9.273416419250881e-19])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_x_s12g_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_s12g", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-3.161091739241630e-05, 0.000000000000000e+00, 0.000000000000000e+00, -9.611287157801698e-03, 0.000000000000000e+00, 0.000000000000000e+00, -2.060278947454855e-01, 0.000000000000000e+00, 0.000000000000000e+00, -3.767659461166153e+00, 0.000000000000000e+00, 0.000000000000000e+00, -3.740187813202134e+00, 0.000000000000000e+00, 0.000000000000000e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
