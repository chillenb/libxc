
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_gga_xc_apf_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_apf", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.446695893280646e+00, -1.036456665932308e+00, -3.282186808955151e-01, -1.387813646805041e-01, -6.378843970803189e-02, -4.939777474977737e-02, -1.605515114320676e-02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_gga_xc_apf_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_apf", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-1.844180141727900e+00, -1.845689325419244e+00, -1.275082412657565e+00, -1.276035721584733e+00, -3.106595083852617e-01, -3.107041193526898e-01, -1.821773255000619e-01, -1.221321552491149e-01, -6.540123248540264e-02, 3.286575301658280e-01, -2.414612260338315e-02, -2.411488988639413e-02, -2.478276820564286e-03, -2.113479979840770e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_gga_xc_apf_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_apf", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-1.534563138643677e-04, 8.065884877314622e-05, -1.527999321170846e-04, -6.200018568643218e-04, 2.626503396015655e-04, -6.176147466756049e-04, -6.382067597607845e-02, 6.204980379880620e-03, -6.372927676348422e-02, 2.728972400922817e-02, 6.154191507864137e+00, -3.934569987461937e+02, -4.198368926206053e+01, 2.117473487534794e+01, -1.435300772946326e+07, -3.448319294896206e+02, 3.130604659007944e-04, -3.453737585981328e+02, -4.261254675586443e+07, 2.962331397286242e-06, -1.269381761532297e+08]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
