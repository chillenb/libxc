
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_gga_xc_cap0_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_cap0", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.405317858371776e+00, -1.002833952525329e+00, -3.280294648125467e-01, -1.356856430304955e-01, -6.140742656525555e-02, -3.294107304255935e-01, -3.493944196403117e-01]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_gga_xc_cap0_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_cap0", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-1.810595913318326e+00, -1.812052699984600e+00, -1.257766640951110e+00, -1.258693535531286e+00, -2.324500945434953e-01, -2.321765111990749e-01, -1.781788680985734e-01, -2.847536654179546e-02, -6.043896906304613e-02, 4.840413256504372e-01, 7.538824618078986e-02, 7.683031356037640e-02, 4.534777313599582e-02, 4.073258330589037e-02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_gga_xc_cap0_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_cap0", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-1.164048221709830e-04, 8.329454893142254e-05, -1.158699682848350e-04, -4.796112046050305e-04, 2.810799129155384e-04, -4.776112161010285e-04, -1.030129478320733e-01, 8.901853713444798e-03, -1.030834050467675e-01, 1.721143872677948e-01, 5.347651277988340e+00, -4.957751996002785e+03, -4.574557080486329e+01, 2.981599346605975e+01, -3.501765690038437e+08, -4.246720401462087e+03, 5.964968469310306e-04, -4.283507932619184e+03, -1.130551572774908e+09, 5.712770042197029e-06, -3.511514492543508e+09]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
