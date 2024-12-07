
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_c_zpbeint_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_zpbeint", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-6.481250426096591e-02, -4.864026334566786e-02, -4.497037507838732e-03, -1.579304186413713e-02, -5.152615871683619e-03, -2.180468351039820e-04, -1.681738173367795e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_c_zpbeint_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_zpbeint", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-1.170848796139399e-01, -1.169882643958780e-01, -1.043794230834702e-01, -1.043461250622943e-01, -2.216597935807534e-02, -2.191283728877559e-02, -2.352095435255322e-02, -1.014230781700488e-01, -1.181988764224054e-03, 2.522693721547374e+00, -3.827707526704428e-02, 4.008509501535606e-02, -1.978391892536515e-04, -2.903012928114558e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_c_zpbeint_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_zpbeint", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [4.225444288885304e-05, 8.450888577770608e-05, 4.225444288885304e-05, 1.418837093372766e-04, 2.837674186745531e-04, 1.418837093372766e-04, 4.259977108408021e-03, 8.519954216816037e-03, 4.259977108408021e-03, 2.645715970152334e+00, 5.291431940304667e+00, 2.645715970152334e+00, -1.337552727747585e+01, -2.675105455495170e+01, -1.337552727747585e+01, -3.550212006894495e+00, -7.100424013788957e+00, -3.550212006894495e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
