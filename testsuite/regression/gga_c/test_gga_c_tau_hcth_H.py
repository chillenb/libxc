
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_c_tau_hcth_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_tau_hcth", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.239243004174028e-02, -9.842604237141735e-03, -1.821809353332712e-02, -1.531515163880383e-02, -1.881032095254189e-03])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_c_tau_hcth_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_tau_hcth", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-1.659200485766327e-02, -1.642687697824275e-01, 1.473529679622479e-02, -2.025314945643365e-01, 1.993374059398031e-03, -2.125987957952046e-01, -1.660119800965484e-02, 9.688198276187938e-02, -2.398227188559773e-03, 7.341023992550259e-02])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_c_tau_hcth_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_tau_hcth", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([4.113964184434564e-02, 0.000000000000000e+00, -1.054469183457117e+21, -1.198908042218195e-02, 0.000000000000000e+00, -8.186046057109924e+20, -4.972803606060557e-02, 0.000000000000000e+00, -3.857140343828428e+20, -1.726919530782615e-01, 0.000000000000000e+00, 1.373323476507595e+20, -2.504789811965593e-01, 0.000000000000000e+00, 2.027130740512963e+14])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
