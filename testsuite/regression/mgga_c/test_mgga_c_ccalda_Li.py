
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_c_ccalda_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_ccalda", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-9.348338102074970e-02, -8.371471638070770e-02, -4.959804183630557e-02, -1.623891792089766e-03, -9.936863716637719e-03, -6.778618829599305e-03, -1.681734331556010e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_c_ccalda_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_ccalda", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-1.025816829330904e-01, -1.022171651255878e-01, -9.267666804721196e-02, -9.220496632585687e-02, -5.658306486640215e-02, -5.675128914954873e-02, -1.767324787881225e-02, -9.284071615397198e-02, 5.030153920974711e+02, 5.029583358523099e+02, -8.515774648031634e-03, -8.623355645200000e-03, -1.978382597500044e-04, -2.903034122798046e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_ccalda_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_ccalda", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-3.708518535064570e-07, -7.417037069227792e-07, -3.708518535064570e-07, -2.479791261100118e-07, -4.959582523002765e-07, -2.479791261100118e-07, 1.198962750719163e-08, 2.397925466816814e-08, 1.198962750719163e-08, 2.019779391651691e+01, 4.039558783303379e+01, 2.019779391651691e+01, -2.935314184277399e+06, -5.870628368554799e+06, -2.935314184277399e+06, 3.615729876660514e-04, 7.231459753322404e-04, 3.615729876660514e-04, 1.718112593986905e+00, 3.436225187973809e+00, 1.718112593986905e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_ccalda_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_ccalda", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([3.853828059757960e-05, 3.853828058752141e-05, 8.623888431698003e-06, 8.623888432344445e-06, -5.682639147854565e-09, -5.682639134458538e-09, -7.066749442886749e-01, -7.066749442885206e-01, 7.019734986253377e+03, 7.019734981414604e+03, -4.282765736091267e-10, -4.282765736201945e-10, -2.915198795043230e-15, -2.915198794746643e-15])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
