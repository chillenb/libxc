
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_gga_xc_b97_k_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_b97_k", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.055608022976873e+00, -7.824404782519889e-01, -2.464726178097618e-01, -9.608284010478127e-02, -4.855045879805438e-02, -9.781651056206030e-03, -1.593245336547453e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_gga_xc_b97_k_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_b97_k", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-1.241596669354130e+00, -1.242321804592330e+00, -8.566379458992542e-01, -8.569962147908059e-01, -3.227941043340930e-01, -3.230129651697843e-01, -1.224334445241484e-01, -2.200654682540060e-01, -5.242540875666597e-02, -1.311160925659943e-01, -1.278174650523133e-02, -1.294761624589803e-02, -1.530780758095733e-04, -3.793159986130155e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_gga_xc_b97_k_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_b97_k", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-2.501300047410511e-04, 0.000000000000000e+00, -2.495558491077036e-04, -9.193547015085398e-04, 0.000000000000000e+00, -9.174974057946771e-04, -3.162157327839477e-03, 0.000000000000000e+00, -2.968292378270255e-03, -1.805264855540130e+00, 0.000000000000000e+00, -3.836040075587272e+01, -2.789975620801083e+01, 0.000000000000000e+00, -4.595715102288768e+03, 1.472401483073791e-01, 0.000000000000000e+00, 7.775212441581862e-02, 2.089000904024208e+00, 0.000000000000000e+00, -8.043619068250921e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
