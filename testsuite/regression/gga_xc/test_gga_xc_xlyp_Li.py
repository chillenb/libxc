
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_xc_xlyp_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_xc_xlyp", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.862360308222018e+00, -1.346080612697096e+00, -4.233444346592936e-01, -1.608401375103562e-01, -8.231350375690391e-02, -9.757573051794655e-02, -3.872465970392271e-02])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_xc_xlyp_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_xc_xlyp", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-2.301541665109833e+00, -2.303495619015054e+00, -1.584371174943455e+00, -1.585561783090844e+00, -4.564616633998660e-01, -4.568578679028933e-01, -2.048086454392650e-01, -1.074814483014574e-01, -7.362013686696789e-02, -3.584467161545599e-02, -2.856760347332607e-02, -2.876746835289931e-02, -5.386621541744422e-03, -4.737680253362847e-03])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_xc_xlyp_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_xc_xlyp", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-2.857808197852257e-04, 5.222815421851711e-06, -2.849387831703775e-04, -1.057084007884814e-03, 3.646941789248587e-05, -1.054306699649490e-03, -7.748973033211447e-02, 4.773762863586187e-02, -7.724847148038833e-02, -4.710357160532498e+00, 4.596134769453040e+00, -9.612073708426163e+02, -7.906349980947554e+01, 2.356939734329661e+01, -3.501915767570857e+07, -8.384113812969717e+02, 7.936097321777658e-02, -8.399295290213694e+02, -1.039681509669580e+08, 0.000000000000000e+00, -3.097099153299291e+08])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
