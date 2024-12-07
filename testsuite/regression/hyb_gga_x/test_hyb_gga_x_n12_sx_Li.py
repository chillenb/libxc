
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_gga_x_n12_sx_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_x_n12_sx", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.254445763111549e+00, -9.705947261818538e-01, -4.196983619507149e-01, -1.126293804092926e-01, -8.323494360256307e-02, -4.839144893597691e-02, -1.109549029528451e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_gga_x_n12_sx_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_x_n12_sx", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-1.150575430055657e+00, -1.151300171359114e+00, -9.332408241080040e-01, -9.337158101950868e-01, -2.404759898475789e-01, -2.411137875181197e-01, -1.371320426514964e-01, -5.887892375488060e-02, -3.466970944157484e-02, -2.388758141770940e-03, -6.108675846026847e-02, -6.077542051362403e-02, -1.599721488760473e-03, -1.138978404141251e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_gga_x_n12_sx_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_x_n12_sx", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-8.696139222956867e-04, 0.000000000000000e+00, -8.676342393100671e-04, -1.919708592278003e-03, 0.000000000000000e+00, -1.915967820962573e-03, -1.598531876995555e-01, 0.000000000000000e+00, -1.595540791710041e-01, -6.503987393680331e+00, 0.000000000000000e+00, -2.112267608041306e+00, -1.609303861097362e+02, 0.000000000000000e+00, -1.987062760029858e+01, -2.097767526594714e+00, 0.000000000000000e+00, -1.965754333209657e+00, -1.452014298152310e+01, 0.000000000000000e+00, -2.082997220539785e+01]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
