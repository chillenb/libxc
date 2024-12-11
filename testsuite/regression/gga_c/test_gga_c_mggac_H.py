
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_c_mggac_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_mggac", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-3.228650861747501e-02, -2.382855897612306e-02, -1.405210303172670e-02, -6.530229730483655e-04, -1.151968923029933e-09])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_c_mggac_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_mggac", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-3.669047342899417e-02, 5.679502823194562e-01, -4.116543789142404e-02, 3.922565710404070e+01, -3.407958789335010e-02, 3.321707534383676e+01, -3.568765530482096e-03, 1.222549764899561e+00, -7.493434614784175e-09, 5.274198293771584e-08])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_c_mggac_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_mggac", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([7.498669326487103e-03, 1.499733865297421e-02, 7.498669326487103e-03, 7.143094750880250e-03, 1.428618950176050e-02, 7.143094750880250e-03, 4.171604330173173e-02, 8.343208660346346e-02, 4.171604330173173e-02, 3.267361637479833e-01, 6.534723274959670e-01, 3.267361637479833e-01, 4.907302861846317e-03, 9.814605722065889e-03, 4.907302861846317e-03])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
