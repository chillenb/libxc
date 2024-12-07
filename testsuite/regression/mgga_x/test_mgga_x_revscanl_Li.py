
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_revscanl_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_revscanl", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.871323669661477e+00, -1.369901998248573e+00, -3.286291578446273e-01, -1.635195422059603e-01, -7.184056519569061e-02, -9.263559814292565e-03, -7.304891950207407e-05]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_revscanl_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_revscanl", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.129371434785948e+00, -2.132829723409528e+00, -1.573767340565626e+00, -1.574808407712013e+00, -4.635909013777146e-01, -4.634039553401215e-01, -1.957543502475791e-01, -8.896025231512965e-03, -9.908290499170161e-02, "nan", -9.511608443302009e-03, "nan", -5.816409163956666e-04, "nan"]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_revscanl_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_revscanl", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-6.279751354063032e-04, 0.000000000000000e+00, -6.226822653116029e-04, -1.327958156916064e-03, 0.000000000000000e+00, -1.325789826063016e-03, 1.227274353516578e-02, 0.000000000000000e+00, 1.227505756795532e-02, -1.087123884080056e+01, 0.000000000000000e+00, 2.227891804276170e+01, 7.211252344146698e+00, 0.000000000000000e+00, "nan", 2.050916078821438e+01, 0.000000000000000e+00, "nan", 4.131689729486103e+04, 0.000000000000000e+00, "nan"]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_revscanl_Li_2_vlapl():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_revscanl", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vlapl"].flatten()
    ref_tgt = [5.500962188364098e-03, 5.471810202211836e-03, 4.218944530435176e-03, 4.222058188930760e-03, 0.000000000000000e+00, 0.000000000000000e+00, 6.701541038106446e-02, 3.888656679023469e-05, 0.000000000000000e+00, "nan", 4.261418431601411e-05, "nan", 2.162553507283074e-05, "nan"]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_revscanl_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_revscanl", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
