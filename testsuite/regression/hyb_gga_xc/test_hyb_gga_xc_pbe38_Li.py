
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_gga_xc_pbe38_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_pbe38", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.181482525353650e+00, -8.457101167318327e-01, -2.631115213362571e-01, -1.151778073790590e-01, -5.166268170271747e-02, -1.284031235377189e-02, -2.399116861682582e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_gga_xc_pbe38_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_pbe38", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-1.517644435049870e+00, -1.518852194185849e+00, -1.050825769396474e+00, -1.051586937728187e+00, -2.664044735978898e-01, -2.665205956601548e-01, -1.525378586031024e-01, -1.139628681103895e-01, -5.433262833547856e-02, 3.423000561651712e-01, -1.716082518854865e-02, -1.703751632419925e-02, -3.463472054091140e-04, -2.462213759884900e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_gga_xc_pbe38_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_pbe38", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-1.135145389953910e-04, 9.190971700708733e-05, -1.129652648568869e-04, -4.824977180007089e-04, 2.980993506782570e-04, -4.804713261155332e-04, -4.354486079002391e-02, 6.249948659585063e-03, -4.342967566588543e-02, 9.122954650990001e-01, 6.762268918356340e+00, 3.207561641176712e+00, -3.101693437506142e+01, 2.258698854598489e+01, 1.018321134771910e+01, -1.762190956256000e-01, 3.357174600576258e-04, -1.645464488748198e-01, -8.082434296286461e-01, 3.212885779437900e-06, -1.156918007121214e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
