
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_k_vsk_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_vsk", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [1.633120205121686e+01, 8.099018794028924e+00, 1.325749423263792e+00, 1.328567860422878e-01, 3.100492591430618e-02, 3.085221565699772e+00, 1.356897645817025e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_k_vsk_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_vsk", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [2.596043056879850e+01, 2.600806960866781e+01, 1.228195426218962e+01, 1.230343869593257e+01, -2.616694648157736e+00, -2.643100864974683e+00, 2.139116343848649e-01, -3.055986349532151e+00, 7.051730554856006e-03, -1.210226216548897e+00, -3.030967574011485e+00, -3.136785468613899e+00, -1.418446558879734e+00, -1.185586320304162e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_k_vsk_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_vsk", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [2.115264106993962e-03, 0.000000000000000e+00, 2.109571755631324e-03, 6.350193195881568e-03, 0.000000000000000e+00, 6.333809258262323e-03, 2.338697145452294e+00, 0.000000000000000e+00, 2.352201806647907e+00, 2.868134834345414e+00, 0.000000000000000e+00, 7.829870593879785e+04, 9.764509993639925e+01, 0.000000000000000e+00, 2.454376727295869e+09, 6.733537848923694e+04, 0.000000000000000e+00, 6.882615844517056e+04, 8.236143115055804e+09, 0.000000000000000e+00, 2.292358375085788e+10]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
