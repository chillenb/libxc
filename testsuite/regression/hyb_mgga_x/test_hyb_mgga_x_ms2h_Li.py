
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_mgga_x_ms2h_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_ms2h", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.776906427211312e+00, -1.245738918378663e+00, -3.393450565433807e-01, -1.599103748402914e-01, -6.935911891709087e-02, -1.558800213885626e-02, -2.912219568563799e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_mgga_x_ms2h_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_ms2h", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.321124355219558e+00, -2.323289374845740e+00, -1.598020835574635e+00, -1.599656501332651e+00, -3.127247597570076e-01, -3.851028015625940e-01, -2.106382425939863e-01, -1.981857015842723e-02, -8.042771624188638e-02, -6.294252676052432e-04, -2.083722506569550e-02, -2.068722971954683e-02, -4.204210754751641e-04, -2.988811186823406e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_x_ms2h_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_ms2h", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-8.070552125418402e-05, 0.000000000000000e+00, -8.026090566134100e-05, -3.302615956918259e-04, 0.000000000000000e+00, -3.257168976047542e-04, -1.271340550693550e-01, 0.000000000000000e+00, -3.293866640626897e-02, -1.381013278432389e+00, 0.000000000000000e+00, -1.763881940737287e-01, -2.637082586820426e+01, 0.000000000000000e+00, -1.835444120355660e+00, -1.794259117883906e-01, 0.000000000000000e+00, -1.673748022840308e-01, -8.222474362901975e-01, 0.000000000000000e+00, -1.447156905092190e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_x_ms2h_Li_2_vlapl():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_ms2h", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vlapl"].flatten()
    ref_tgt = [0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_x_ms2h_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_ms2h", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [8.488401251594875e-06, 8.228370053850840e-12, 6.040256208660576e-05, 1.206197136996259e-17, 2.234975866223325e-02, 6.905343007255664e-11, 5.640521714617188e-03, 8.878958975883119e-18, 6.099547172096155e-07, 2.876205423475437e-10, 8.383761626389689e-22, 3.163746705027331e-18, 4.441315804982689e-39, 1.178674109284832e-11]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
