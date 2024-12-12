
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_c_revtpss_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_revtpss", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-5.100635995945699e-02, -4.180433668848788e-02, -4.803380872656833e-03, -2.562445064230058e-03, -9.501324022482767e-09, -2.003529827382377e-08, -5.655232412814755e-16])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_c_revtpss_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_revtpss", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-1.070769587121273e-01, -1.070472505496760e-01, -8.958262082833506e-02, -8.956060567932923e-02, -2.108592340735118e-02, -2.120636852737655e-02, -2.984747271384492e-02, -4.403326901085505e-01, -4.144723525424997e-03, -1.513962805831394e-02, -1.273977751910282e-07, -1.305157405461327e-07, -3.577733455464826e-15, -4.233361194955901e-15])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_revtpss_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_revtpss", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([8.418488127426429e-05, 1.895928374572110e-04, 8.421024257226921e-05, 2.137965090540415e-04, 2.954338226692877e-04, 2.136633899037879e-04, 2.166594423453821e-02, -3.100189249575413e-02, 2.172967795000078e-02, 3.378488477441667e+01, 8.734989155463344e+01, 8.281375372280158e+02, 2.418546191004956e+01, 4.833712340946774e+01, 9.922133218846310e+04, 1.216748561778252e-03, -7.009599139035285e-04, 1.251483834404895e-03, 5.059265615691109e-06, 1.011868011772730e-05, 5.059265615716128e-06])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_revtpss_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_revtpss", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([-7.171047840384917e-03, -7.171047840384915e-03, -3.251987149093205e-03, -3.251987149093204e-03, 6.975164119967073e-04, 6.975164119967066e-04, -1.169741281697824e+00, -1.169741281697566e+00, -5.783869041154399e-02, -5.783869036481573e-02, 8.468433496463786e-14, 8.468433496463785e-14, -2.845928495864766e-31, -2.845928495864767e-31])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
