
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_revscanl_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_revscanl", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-2.037422781146677e+00, -1.413111899517623e+00, -3.146632681628517e-01, -1.841757577255072e-01, -7.184056513423433e-02, -5.333840596520673e-03, -2.726525612367122e-05])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_revscanl_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_revscanl", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-2.719573523920549e+00, -2.722014612766085e+00, -1.892035111043257e+00, -1.893636405881253e+00, -3.781661776080784e-02, -1.464774335915488e-01, -2.457883112651348e-01, -8.782769488810807e-03, -9.908290499170161e-02, -8.622523944629394e-05, -1.017496767136451e-02, -9.250416365297357e-03, -6.246199011163051e-04, 7.898955841011051e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_revscanl_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_revscanl", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([7.260008312118194e-06, 0.000000000000000e+00, 7.224577549782230e-06, 4.578689914843170e-05, 0.000000000000000e+00, 4.561034488914347e-05, -4.776887504482442e-01, 0.000000000000000e+00, -3.439492758716105e-01, 6.865366266802722e-02, 0.000000000000000e+00, 2.206877992259312e+01, 7.211252344146698e+00, 0.000000000000000e+00, 2.070060342793121e+04, 2.692142761504849e+01, 0.000000000000000e+00, 1.981030296594340e+01, 4.131763925328588e+04, 0.000000000000000e+00, -2.761938068797651e+07])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_revscanl_Li_2_vlapl():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_revscanl", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vlapl"].flatten()
    ref_tgt = numpy.asarray([0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 2.235665979961642e-02, 1.623207593454198e-02, 0.000000000000000e+00, -3.244525982637811e-07, 0.000000000000000e+00, 7.609881204011699e-09, -4.210539923672998e-07, -3.145498333090593e-07, -1.777821361908888e-15, 3.917042626942934e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
