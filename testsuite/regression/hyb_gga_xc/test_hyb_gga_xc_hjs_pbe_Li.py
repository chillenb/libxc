
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_gga_xc_hjs_pbe_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_hjs_pbe", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.421256650110743e+00, -1.021641031258458e+00, -3.304970721287947e-01, -1.487658149704318e-01, -7.443193449677817e-02, -2.045988379791419e-02, -3.838580991593412e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_gga_xc_hjs_pbe_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_hjs_pbe", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-1.813318558188770e+00, -1.814793124293703e+00, -1.256116656063051e+00, -1.257048676497882e+00, -3.316184477176310e-01, -3.317543597685651e-01, -1.918722790752707e-01, -1.235974816351044e-01, -7.574959545323719e-02, 3.419889436340118e-01, -2.726464243052726e-02, -2.707265921140882e-02, -5.541542701095840e-04, -3.939536531630567e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_gga_xc_hjs_pbe_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_hjs_pbe", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-1.456761478558856e-04, 9.190971700708733e-05, -1.450160762011434e-04, -6.093140305489382e-04, 2.980993506782570e-04, -6.068824327063247e-04, -5.304435699337283e-02, 6.249948659585063e-03, -5.290719326757158e-02, 1.084603245813662e-01, 6.762268918356340e+00, 3.107033530498697e+00, -4.448745346219213e+01, 2.258698854598489e+01, 9.517923380242198e+00, -2.778344993649839e-01, 3.357174600576258e-04, -2.595332227624224e-01, -1.292669611253401e+00, 3.212885779437900e-06, -1.850346223145298e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
