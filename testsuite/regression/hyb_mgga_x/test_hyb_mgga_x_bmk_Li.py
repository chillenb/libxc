
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_mgga_x_bmk_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_bmk", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-9.541037376549746e-01, -7.534538000923729e-01, -2.051716907556120e-01, -8.057219847798210e-02, -4.747460117418048e-02, -4.013576367901429e-02, -7.543110707328327e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_mgga_x_bmk_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_bmk", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-8.446441182825496e-01, -8.457732448253330e-01, -6.650641902504656e-01, -6.650415667352695e-01, -2.251288794893714e-01, -2.250486789235636e-01, -8.617134842297458e-02, -5.049427547086364e-02, -5.356381490827040e-02, -1.630252094781789e-03, -5.298890391615717e-02, -5.265371761361910e-02, -1.088945286981023e-03, -7.741458841264762e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_x_bmk_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_bmk", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-7.698661911125948e-04, 0.000000000000000e+00, -7.675251204943623e-04, -1.835141475322624e-03, 0.000000000000000e+00, -1.832563618640785e-03, -9.035307185012748e-02, 0.000000000000000e+00, -9.296703883563387e-02, -1.270897145834668e+01, 0.000000000000000e+00, -5.823608507243606e+00, -4.420475557317642e+01, 0.000000000000000e+00, -3.742655348235190e+01, -5.914839124375774e+00, 0.000000000000000e+00, -5.524791879398721e+00, -2.724535260416402e+01, 0.000000000000000e+00, -3.899898912951645e+01]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_x_bmk_Li_2_vlapl():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_bmk", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vlapl"].flatten()
    ref_tgt = [0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_x_bmk_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_bmk", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [3.151499998458765e-03, 3.179795732437428e-03, 1.334554802018818e-03, 1.319973791269397e-03, 2.531149466731900e-02, 2.645111214175963e-02, 1.377488794617625e-01, -1.843058759560192e-09, 8.762694511951515e-02, -1.019542107248507e-15, -2.064470225081858e-14, -2.103771492599221e-09, 6.880430686300299e-25, -2.614562097528341e-17]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
