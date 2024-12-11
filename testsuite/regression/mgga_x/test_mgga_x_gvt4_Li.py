
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_gvt4_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_gvt4", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.900737744303900e+00, -1.305925424998269e+00, 9.480440824327085e-02, -1.722526013862984e-01, -4.293099434239726e-02, -1.236317973756253e-04, 3.414588393735454e-08]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_gvt4_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_gvt4", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.558645743690247e+00, -2.560948493090017e+00, -1.795211560626460e+00, -1.796620544346122e+00, -9.891002384247912e-01, -9.711562126913741e-01, -2.308637704898440e-01, -6.799862745921147e-04, -1.305807190451011e-01, -5.539174035671294e-08, -1.647538110302280e-05, -7.531293293296333e-04, -1.115227109678260e-13, 3.875472823380391e-07]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_gvt4_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_gvt4", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-6.715437371220428e-05, 0.000000000000000e+00, -6.680988295523206e-05, -4.041603384933001e-04, 0.000000000000000e+00, -4.022051573361446e-04, -1.341531559666212e+00, 0.000000000000000e+00, -1.309052136149183e+00, -8.555286184251065e-01, 0.000000000000000e+00, 5.124966859315992e-01, -2.774610374989784e+02, 0.000000000000000e+00, 3.446343561952663e+00, 2.293355829543077e-04, 0.000000000000000e+00, 4.848095952365034e-01, 1.573235012845672e-10, 0.000000000000000e+00, -3.352687020214406e+04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_gvt4_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_gvt4", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [3.631200375184863e-03, 3.625312009662809e-03, 8.007463231530123e-03, 7.987252432947945e-03, 2.046116881433016e-01, 2.024650137487596e-01, 3.178711427376042e-02, 6.521970465712590e-05, 6.140683731833247e-01, 1.338424571260003e-08, 3.249316640851868e-08, 7.038473453363094e-05, 1.820531609116478e-19, -1.089597390377134e-07]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
