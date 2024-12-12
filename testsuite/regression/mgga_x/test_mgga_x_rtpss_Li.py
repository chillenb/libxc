
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_rtpss_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_rtpss", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.989315410805445e+00, -1.381379396531988e+00, -4.280240465549673e-01, -1.784991558882010e-01, -7.889313579225732e-02, -2.055687026651119e-02, -3.838588870220200e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_rtpss_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_rtpss", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-2.395452694425670e+00, -2.397596019767925e+00, -1.704574342960816e+00, -1.706352813565569e+00, -3.028645350227939e-01, -3.039037246128635e-01, -2.122560827762761e-01, -2.615912580860028e-02, -6.876699995344754e-02, -8.296468243503926e-04, -2.750809938117428e-02, -2.730803076839329e-02, -5.541564195188991e-04, -3.939545845223386e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_rtpss_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_rtpss", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-1.167502387942236e-03, 0.000000000000000e+00, -1.166579173294966e-03, -1.865624458098191e-03, 0.000000000000000e+00, -1.862617620322369e-03, -1.318887275480556e-01, 0.000000000000000e+00, -1.309600115266877e-01, -3.280480004639443e+01, 0.000000000000000e+00, 0.000000000000000e+00, -8.409719776511399e+01, 0.000000000000000e+00, 0.000000000000000e+00, -1.076118101077889e-308, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_rtpss_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_rtpss", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([6.052249577660092e-02, 6.067826877462670e-02, 3.183378676741723e-02, 3.193284846878300e-02, 7.705768618366577e-04, 6.473760624367904e-04, 1.183655197960039e+00, 0.000000000000000e+00, 1.681035068691340e-02, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
