
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_mgga_xc_br3p86_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_br3p86", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.186076352325020e+00, -9.945875343870554e-01, -3.599507717358149e-01, -1.318454243042449e-01, -6.860081466527569e-02, -1.207026815845985e-01, -3.032478076048748e-02])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_mgga_xc_br3p86_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_br3p86", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-1.623111139421622e+00, -1.624404104913967e+00, -1.442995714877170e+00, -1.444128596738197e+00, -3.435409571992629e-01, -3.434482611770217e-01, -1.957837139354236e-01, -1.821221751702601e-01, -7.811199807554900e-02, -7.396453482586732e-02, -6.138611477647668e-02, -6.125639967755888e-02, -2.625912161880091e-03, -1.333839352248739e-03])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_xc_br3p86_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_br3p86", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([3.657229280797616e-05, 7.418007355492812e-05, 3.657230527631286e-05, -7.029647006493383e-04, 2.495232493095846e-04, -7.007244143367059e-04, -6.177802233728919e-02, 1.143008717544878e-02, -6.176176212622205e-02, 5.029557179507135e-01, 4.457961088867724e+00, -6.332850844500309e+02, -2.860802511087190e+01, 4.963235437101455e+01, -3.095892980318790e+07, -5.592589666361841e+02, -8.145842987820048e-03, -5.649462756080887e+02, -9.643108972339182e+07, -1.784454114815711e-29, -2.954630466937379e+08])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_xc_br3p86_Li_2_vlapl():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_br3p86", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vlapl"].flatten()
    ref_tgt = numpy.asarray([-6.716295533922670e-06, -6.734304406971326e-06, -3.593596708983498e-03, -3.593029391035820e-03, -4.062512428291185e-03, -4.056354137826972e-03, -1.656433105758324e-02, -2.029148655650437e-03, -3.194067606861329e-02, -3.153443627006714e-03, -2.076397825875842e-03, -2.052079528009510e-03, -2.927070607472619e-03, -3.222260641112032e-03])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_xc_br3p86_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_br3p86", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([2.686518213569068e-05, 2.693721762788530e-05, 1.437438683593396e-02, 1.437211756414328e-02, 1.625004971316474e-02, 1.622541655130789e-02, 6.625732423033295e-02, 8.116594622601749e-03, 1.277627042744532e-01, 1.261377450802686e-02, 8.305591303503368e-03, 8.208318112038038e-03, 1.170828242989048e-02, 1.288904256444813e-02])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
