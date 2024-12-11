
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_ms2_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ms2", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.890839356852569e+00, -1.244813783515584e+00, -2.816370634950895e-01, -1.732214932593697e-01, -6.085920072318025e-02, -1.713407509232874e-02, -2.969801431810358e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_ms2_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ms2", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.653624472048424e+00, -2.656065991614337e+00, -1.872550613349514e+00, -1.874669671912180e+00, -3.770128478203744e-01, -3.769502350959603e-01, -2.358292291276765e-01, -2.177864852574421e-02, -8.238217305166402e-02, -6.916765007599846e-04, -2.293285346850760e-02, -2.273321947202948e-02, -4.620018047520339e-04, -2.121976006489333e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_ms2_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ms2", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-4.746092955001893e-04, 0.000000000000000e+00, -4.726640732781295e-04, -2.394465460166750e-03, 0.000000000000000e+00, -2.391007830937139e-03, -2.220771124620191e-01, 0.000000000000000e+00, -2.224828455249058e-01, -5.035591340065741e+00, 0.000000000000000e+00, -1.938331803008008e-01, -1.061844801239717e+02, 0.000000000000000e+00, -1.241225140869092e+00, -8.267744226796889e-05, 0.000000000000000e+00, -1.839283541582756e-01, 2.097315329325820e-08, 0.000000000000000e+00, -2.165974460106247e+12]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_ms2_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ms2", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [1.906264285589711e-02, 1.903376668714959e-02, 3.124389139770947e-02, 3.130966702703184e-02, 2.600034094430618e-04, 2.794145058561458e-04, 1.364902935899198e-01, 9.757097775695735e-18, 1.035366479497542e-02, 1.416170360379346e-18, -1.595291091008242e-20, 3.476644730799265e-18, -2.553356255305009e-18, 3.388948450895370e-19]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
