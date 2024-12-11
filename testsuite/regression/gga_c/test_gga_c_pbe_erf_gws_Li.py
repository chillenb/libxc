
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_c_pbe_erf_gws_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_pbe_erf_gws", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-5.723873479144241e-02, -4.125521695121536e-02, -5.048854861858292e-03, -2.172655147041317e-03, -5.861716530430343e-05, -1.141628619987541e-05, -5.041102820139185e-11])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_c_pbe_erf_gws_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_pbe_erf_gws", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-1.098442340448478e-01, -1.097228740077402e-01, -9.446835784148656e-02, -9.437827201153039e-02, -2.119436711039607e-02, -2.120178124067217e-02, -4.427824752877531e-03, -3.095066844260547e-02, -1.461414560099423e-04, -3.370118460745902e-03, -2.277511945974944e-05, -2.326706310468423e-05, -6.852340482379549e-11, -1.907193707619031e-10])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_c_pbe_erf_gws_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_pbe_erf_gws", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([4.238591483309817e-05, 8.477182966619633e-05, 4.238591483309817e-05, 1.351820703328113e-04, 2.703641406656226e-04, 1.351820703328113e-04, 4.195074974435452e-03, 8.390149948870903e-03, 4.195074974435452e-03, 1.164999850480891e-02, 2.329999700961782e-02, 1.164999850480891e-02, 8.138892616977220e-05, 1.627778523395444e-04, 8.138892616977220e-05, 1.503230548497418e-03, 3.006461096994836e-03, 1.503230548497418e-03, 5.345968980378207e-07, 1.069193796075641e-06, 5.345968980378207e-07])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
