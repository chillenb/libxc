
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_x_optb88_vdw_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_optb88_vdw", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.789781105497965e+00, -1.274825708745828e+00, -4.086216391230124e-01, -1.598791251652218e-01, -7.798897094991206e-02, -1.159758941648124e-01, -4.610685898715405e-02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_x_optb88_vdw_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_optb88_vdw", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.255220092324656e+00, -2.257335252823017e+00, -1.538690496770865e+00, -1.540053626272181e+00, -3.439275264074395e-01, -3.437888651727409e-01, -2.058482254177222e-01, -3.265251948050180e-02, -7.452249775509283e-02, -6.700596473818025e-03, -3.347213768413099e-02, -3.361038443682237e-02, -6.454101579909755e-03, -5.575687921099839e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_x_optb88_vdw_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_optb88_vdw", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-2.230046861680168e-04, 0.000000000000000e+00, -2.222633778696230e-04, -8.450399860671228e-04, 0.000000000000000e+00, -8.423822360877126e-04, -9.741274096463198e-02, 0.000000000000000e+00, -9.738815865203837e-02, -3.573509672964027e+00, 0.000000000000000e+00, -1.151636328246177e+03, -6.447147492708997e+01, 0.000000000000000e+00, -4.168561613969085e+07, -1.001447638685450e+03, 0.000000000000000e+00, -1.003036860942780e+03, -1.237596228089616e+08, 0.000000000000000e+00, -3.686661327772879e+08]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
