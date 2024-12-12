
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_rmsrpbel_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_rmsrpbel", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.893861794430904e+00, -1.336002730733107e+00, -4.133469764920200e-01, -1.700052311336628e-01, -7.935492430695917e-02, -2.055687026651119e-02, -3.838588870220200e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_rmsrpbel_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_rmsrpbel", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-2.443127529200996e+00, -2.445425539724186e+00, -1.669970569079717e+00, -1.671641627055910e+00, -2.617597236414930e-01, -3.585007985232939e-01, -2.223630887328126e-01, -2.615912580860030e-02, -7.739192810946098e-02, -8.296468243503833e-04, -2.750809938117428e-02, -2.730803076839329e-02, -5.541564195188991e-04, -3.939545845223386e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_rmsrpbel_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_rmsrpbel", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-1.387401052317862e-04, 0.000000000000000e+00, -1.381264977892916e-04, -5.845697177589410e-04, 0.000000000000000e+00, -5.799379079595347e-04, -2.204676147855114e-01, 0.000000000000000e+00, -9.376690422409048e-02, -2.200172100972504e+00, 0.000000000000000e+00, -3.092674806204462e-192, -6.217805458803848e+01, 0.000000000000000e+00, 0.000000000000000e+00, -5.342829531239033e-172, 0.000000000000000e+00, -6.754522588623127e-181, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_rmsrpbel_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_rmsrpbel", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([6.294134491154155e-06, 6.100854992123767e-12, 4.668814260095217e-05, 9.322422716532268e-18, 3.018429450433036e-02, 9.370651767508756e-11, 4.108792241739361e-03, 0.000000000000000e+00, 6.609981572566337e-07, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
