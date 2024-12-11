
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_gga_xc_pbe0_13_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_pbe0_13", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.256248771460716e+00, -8.991934294987469e-01, -2.804470457789002e-01, -1.218454979279901e-01, -5.501698430987764e-02, -1.369633267004336e-02, -2.559057985794635e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_gga_xc_pbe0_13_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_pbe0_13", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-1.611086577407060e+00, -1.612383242358196e+00, -1.114117066416272e+00, -1.114935278634071e+00, -2.831030225474805e-01, -2.832265088839606e-01, -1.610924519905581e-01, -1.150510237465210e-01, -5.751600371561040e-02, 3.422654876934818e-01, -1.830487691782816e-02, -1.817334744602883e-02, -3.694370191029793e-04, -2.626361343876331e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_gga_xc_pbe0_13_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_pbe0_13", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-1.241458321619867e-04, 9.190971700708733e-05, -1.235599397475823e-04, -5.246008775566982e-04, 2.980993506782570e-04, -5.224393928791775e-04, -4.665618313134500e-02, 6.249948659585063e-03, -4.653331899893064e-02, 7.477061988270552e-01, 6.762268918356340e+00, 3.195990119976615e+00, -3.383762961826501e+01, 2.258698854598489e+01, 1.010919248603421e+01, -1.879782259159751e-01, 3.357174600576258e-04, -1.755274027151429e-01, -8.621264320401284e-01, 3.212885779437900e-06, -1.234045981365534e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
