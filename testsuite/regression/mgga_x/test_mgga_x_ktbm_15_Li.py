
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_ktbm_15_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_15", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.955350305992567e+00, -1.287569444566700e+00, -2.853873470131611e-01, -1.807126731644930e-01, -5.887379976927462e-02, -1.219409072973294e-02, -2.277135609801538e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_ktbm_15_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_15", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.789430296265298e+00, -2.792164207873859e+00, -1.919432964778287e+00, -1.921239133745192e+00, -3.523707833477764e-01, -3.515307573478215e-01, -2.519103238865433e-01, -1.487434568138115e-02, -7.482557502871176e-02, -4.717454185219262e-04, -1.564137846170520e-02, -1.552762526373434e-02, -3.150988397550709e-04, -2.337871375588938e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_ktbm_15_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_15", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-6.665134520824165e-04, 0.000000000000000e+00, -6.642708893631630e-04, -2.450233296180653e-03, 0.000000000000000e+00, -2.443888986179463e-03, -2.998019775696890e-02, 0.000000000000000e+00, -3.181217760368189e-02, -1.052816355034690e+01, 0.000000000000000e+00, -1.651191071355251e+01, -6.310003667364956e+01, 0.000000000000000e+00, -4.131918029226507e+04, -3.067837823999398e-01, 0.000000000000000e+00, -1.476441746897872e+01, -6.256890031118211e-01, 0.000000000000000e+00, 2.066135517428939e+05]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_ktbm_15_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_15", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [2.617168143390780e-02, 2.615266733726960e-02, 2.976572561542134e-02, 2.978045701316581e-02, -5.178340283719220e-03, -5.333163982714227e-03, 3.096744720690774e-01, 2.111359480655077e-04, -3.075255775803402e-02, 1.683494838866313e-05, 4.556223024835578e-06, 2.147901396954057e-04, 7.596869000226326e-11, -1.331730350303782e-11]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
