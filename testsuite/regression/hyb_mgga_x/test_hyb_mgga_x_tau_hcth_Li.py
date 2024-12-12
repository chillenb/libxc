
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_mgga_x_tau_hcth_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_tau_hcth", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.525659448450171e+00, -1.079981953781580e+00, -4.039377057091086e-01, -1.370713717899547e-01, -6.780044561801035e-02, -2.896053752363114e-02, -5.418168207094333e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_mgga_x_tau_hcth_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_tau_hcth", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-1.967418352320598e+00, -1.969315724092660e+00, -1.318454462291565e+00, -1.319640837308258e+00, -1.435573290807838e-01, -1.414183002575757e-01, -1.799881227348233e-01, -3.672928602166878e-02, -4.139034546809670e-02, -1.171032597309659e-03, -3.859972639514878e-02, -3.832997588001615e-02, -7.821890141712753e-04, -5.560660498441325e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_x_tau_hcth_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_tau_hcth", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-1.173585732788481e-04, 0.000000000000000e+00, -1.168591704768317e-04, -6.404358837100165e-04, 0.000000000000000e+00, -6.382173526195598e-04, -1.992886909228825e-01, 0.000000000000000e+00, -2.006697877900722e-01, -1.515816673556191e+00, 0.000000000000000e+00, -1.244966432321287e+00, -1.043149819016673e+02, 0.000000000000000e+00, -7.977681503940691e+00, -1.264887555230684e+00, 0.000000000000000e+00, -1.181299432509990e+00, -5.807474257710752e+00, 0.000000000000000e+00, -8.312807558285295e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_x_tau_hcth_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_tau_hcth", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([3.669653352051378e-04, 3.692142167767331e-04, 9.068671118897779e-05, 8.842444545084275e-05, 2.954976939577847e-03, 3.129267403169649e-03, 8.335269201541146e-03, 2.710184102798595e-08, -1.121462871226628e-02, 1.412260888820372e-14, 3.057468317737517e-13, 3.105391039183646e-08, -9.529841621050820e-24, 3.621257207865554e-16])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
