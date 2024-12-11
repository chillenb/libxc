
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_revtm_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_revtm", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.750487057405643e+00, -1.209497171408433e+00, -2.970707286254540e-01, -1.585321081890510e-01, -6.338278030754924e-02, -6.558799535747394e-02, -2.607676155272054e-02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_revtm_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_revtm", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.348217241049760e+00, -2.350329526671797e+00, -1.627778173544259e+00, -1.629195353418353e+00, -3.932362770733530e-01, -3.929112863240867e-01, -2.123117975182650e-01, -2.085620783475743e-02, -8.285624964998076e-02, -2.445477128912467e-03, -5.244592009531068e-02, -2.155148277148766e-02, -2.073793546424148e-02, -2.252801297685577e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_revtm_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_revtm", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-4.230120387706564e-04, 0.000000000000000e+00, -4.216093562300873e-04, -1.063031436655224e-03, 0.000000000000000e+00, -1.061365833021507e-03, -2.172229377305118e-01, 0.000000000000000e+00, -2.170666390923392e-01, -9.547921158937022e+00, 0.000000000000000e+00, -5.047770576772503e+02, -8.971595141189715e+01, 0.000000000000000e+00, -3.497866012664654e+06, -1.673191940199520e+01, 0.000000000000000e+00, -4.456805425096432e+02, -6.734408188451059e+02, 0.000000000000000e+00, -2.313234475711269e+12]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_revtm_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_revtm", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [2.212851959677002e-03, 2.186650153183742e-03, 2.312447242709724e-03, 2.316267962500452e-03, -5.398431828091472e-04, -5.718172461599672e-04, 2.512507203979862e-02, 3.895036744553971e-03, -1.385004212458444e-02, 7.367690545649392e-04, 1.285571263862510e-04, 3.912939835595237e-03, 4.225969770970404e-08, -4.538560600103215e-13]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
