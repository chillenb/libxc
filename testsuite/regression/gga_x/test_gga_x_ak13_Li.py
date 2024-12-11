
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_x_ak13_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_ak13", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.817134242906705e+00, -1.330215456382148e+00, -7.412624109362473e-01, -1.616799694554466e-01, -1.099693803250125e-01, -2.584645771598166e+00, -3.806177884673261e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_x_ak13_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_ak13", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-2.173266467510051e+00, -2.175506857205953e+00, -1.399422316964973e+00, -1.400866558895327e+00, 1.517807472306548e-01, 1.533350933569954e-01, -2.023354435284849e-01, 1.086924662613290e+00, -1.926419028851911e-02, 7.643826750853210e-01, 1.077886936153048e+00, 1.098654851986726e+00, 8.342377268720956e-01, 7.655674695898633e-01])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_x_ak13_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_ak13", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-4.260013986699590e-04, 0.000000000000000e+00, -4.243745213352998e-04, -1.967994809060353e-03, 0.000000000000000e+00, -1.961130139222232e-03, -5.526078167912231e-01, 0.000000000000000e+00, -5.532739458195578e-01, -5.913349392495406e+00, 0.000000000000000e+00, -4.383202189376836e+04, -2.786953657787420e+02, 0.000000000000000e+00, -3.910592234013786e+09, -3.732176720585583e+04, 0.000000000000000e+00, -3.773761428158284e+04, -1.300977921487786e+10, 0.000000000000000e+00, -4.102125784351829e+10])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
