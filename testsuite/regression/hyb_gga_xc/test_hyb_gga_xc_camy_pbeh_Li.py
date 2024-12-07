
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_gga_xc_camy_pbeh_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_camy_pbeh", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-3.009343857658015e-01, -2.812157173735934e-01, -2.222037394424309e-01, -1.162553008161714e-01, -6.301627793964878e-02, -1.641676725914905e-02, -3.070868289721909e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_gga_xc_camy_pbeh_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_camy_pbeh", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-3.438836635543532e-01, -3.438151155693305e-01, -3.036534161282132e-01, -3.036109353677126e-01, -1.833580260135325e-01, -1.835643680076815e-01, -1.422908617087371e-01, -1.185003935651108e-01, -6.254523050350366e-02, 3.421548696289703e-01, -2.192779002623188e-02, -2.177077775623357e-02, -4.433241115439379e-04, -3.151632493903499e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_gga_xc_camy_pbeh_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_camy_pbeh", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [7.243316348232825e-06, 9.190971700708733e-05, 7.440240969271145e-06, -1.084042426416597e-04, 2.980993506782570e-04, -1.073516360138294e-04, -4.036729379100196e-02, 6.249948659585063e-03, -4.024486165657689e-02, 5.132092093175000e-01, 6.762268918356340e+00, 3.159031321505296e+00, -4.259641346691586e+01, 2.258698854598489e+01, 9.872332144179998e+00, -2.255262344193490e-01, 3.357174600576258e-04, -2.105904590224234e-01, -1.034552035890164e+00, 3.212885779437900e-06, -1.480855497076789e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
