
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_mspbel_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_mspbel", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.921759772184896e+00, -1.351987934384125e+00, -3.922000832707392e-01, -1.726975774278277e-01, -7.781158654272440e-02, -2.053487569831258e-02, -3.838585506945526e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_mspbel_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_mspbel", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.492634637745184e+00, -2.494970393257419e+00, -1.709730347241210e+00, -1.711460924388483e+00, -3.045934361137848e-01, -3.952269923655934e-01, -2.265641893988986e-01, -2.608213483510489e-02, -8.327350724404210e-02, -8.296399035302603e-04, -2.741778485069950e-02, -2.722272306143002e-02, -5.541548354971549e-04, -3.939537905292981e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_mspbel_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_mspbel", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-1.176249863076902e-04, 0.000000000000000e+00, -1.170625321143571e-04, -4.876572070686869e-04, 0.000000000000000e+00, -4.828199852394270e-04, -1.807388734848887e-01, 0.000000000000000e+00, -6.227206346761185e-02, -1.920134405834805e+00, 0.000000000000000e+00, -4.925669667953466e-01, -4.480488385715350e+01, 0.000000000000000e+00, -4.559681659508071e+00, -5.010438138482686e-01, 0.000000000000000e+00, -4.673562674436461e-01, -2.299384621547234e+00, 0.000000000000000e+00, -5.478029211774444e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_mspbel_Li_2_vlapl():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_mspbel", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vlapl"].flatten()
    ref_tgt = [0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_mspbel_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_mspbel", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [7.735181274308015e-06, 7.497925142775974e-12, 5.624475731796758e-05, 1.123113841769123e-17, 2.818731089760575e-02, 8.755441975682267e-11, 5.092642960794124e-03, 1.759412844824860e-17, 6.776344361014923e-07, 5.708342219331681e-10, 6.643981095677627e-21, 6.268541651888014e-18, -3.675026236653643e-37, 9.539065721105236e-11]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
