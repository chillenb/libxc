
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_mgga_x_revm06_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_revm06", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-9.112128923682192e-01, -6.898118763221014e-01, -1.270550660252904e-01, -8.012807872106792e-02, -3.524322459844006e-02, -3.715554869996945e-02, -6.326937597134115e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_mgga_x_revm06_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_revm06", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-9.875311015171984e-01, -9.887424536378471e-01, -8.372796702584431e-01, -8.376333613985972e-01, -2.752550652822724e-01, -2.785090001614382e-01, -1.162430070521372e-01, -4.681237847836917e-02, -5.434974403259250e-02, -1.503773137099807e-03, -4.984676928190965e-02, -4.882971907006513e-02, -1.004466745857233e-03, -3.958242799218884e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_x_revm06_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_revm06", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-1.547682794086494e-04, 0.000000000000000e+00, -1.541674736621180e-04, -7.461148499362215e-04, 0.000000000000000e+00, -7.433450558378309e-04, -3.086603955072464e-01, 0.000000000000000e+00, -3.085790454825338e-01, -2.222622049815902e+00, 0.000000000000000e+00, -9.594334850522954e-01, -1.352993315919682e+02, 0.000000000000000e+00, -6.164878852395131e+00, -4.106111925024236e-04, 0.000000000000000e+00, -9.102141267466308e-01, -2.813910393102387e-10, 0.000000000000000e+00, -6.347907882776851e+12])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_x_revm06_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_revm06", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([-3.227657677360535e-02, -3.222588512464787e-02, -1.201105584722742e-02, -1.205929407082600e-02, 1.941266935197311e-02, 2.013758427739465e-02, 2.670181246559369e-01, -4.940325382679366e-05, 6.161475321776420e-02, -1.012454772001872e-08, -2.458023059598519e-08, -5.332018560703878e-05, -1.377481943704725e-19, -2.662485377689952e-09])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
