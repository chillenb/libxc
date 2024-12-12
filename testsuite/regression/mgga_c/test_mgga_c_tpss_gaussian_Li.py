
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_c_tpss_gaussian_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_tpss_gaussian", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-4.934998883632613e-02, -4.020232965266050e-02, -4.001281559425683e-03, -2.468452642756615e-03, -6.150191457942031e-09, -7.610292449094138e-09, -1.790424362528482e-16])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_c_tpss_gaussian_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_tpss_gaussian", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-1.055073295579365e-01, -1.054747853907534e-01, -8.828604941806428e-02, -8.826079265669380e-02, -1.836370869599673e-02, -1.845907485756521e-02, -2.922093050542940e-02, -3.961290088760774e-01, -2.696198933090156e-03, -9.667915969275612e-03, -4.883755371422192e-08, -4.993138184521617e-08, -1.135758626656430e-15, -1.343644341733287e-15])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_tpss_gaussian_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_tpss_gaussian", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([8.317711165706782e-05, 1.933547508966203e-04, 8.320973591704655e-05, 2.081294986967723e-04, 3.188333859070188e-04, 2.080357672027582e-04, 1.724627919416701e-02, -2.337011064695352e-02, 1.729639531707931e-02, 3.293810122958886e+01, 8.871827059292728e+01, 4.555383480053485e+02, 1.573295984388997e+01, 3.144461526301395e+01, 6.253977066112035e+04, 4.337572391372139e-04, -2.079030226697707e-04, 4.456754269432319e-04, 1.606375526063325e-06, 3.212751052298149e-06, 1.606375526071368e-06])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_tpss_gaussian_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_tpss_gaussian", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([-7.228876034607547e-03, -7.228876034607546e-03, -3.378081523828133e-03, -3.378081523828132e-03, 5.394847490658007e-04, 5.394847490658004e-04, -1.137149538100346e+00, -1.137149538100096e+00, -3.762481170314210e-02, -3.762481167274474e-02, 2.604408427955354e-14, 2.604408427955356e-14, -9.070442358332176e-32, -9.070442358332180e-32])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
