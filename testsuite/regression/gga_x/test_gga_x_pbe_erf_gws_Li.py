
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_x_pbe_erf_gws_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_pbe_erf_gws", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.526369656933796e+00, -1.018556771534113e+00, -1.379098388065521e-01, -2.392410862484068e-02, -1.804583783000959e-03, -1.152338488304501e-05, -7.921037573910535e-11])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_x_pbe_erf_gws_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_pbe_erf_gws", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-1.976514898212918e+00, -1.978625343082763e+00, -1.266233656709832e+00, -1.267566748808146e+00, -1.922307884012761e-01, -1.921894972067527e-01, -4.453597311196764e-02, -2.002848115565989e-05, -3.561897864350223e-03, -6.399969837504244e-10, -2.328541136942439e-05, -2.278163293888121e-05, -1.907197500762611e-10, -6.852313113021059e-11])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_x_pbe_erf_gws_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_pbe_erf_gws", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-2.435785486069672e-04, 0.000000000000000e+00, -2.427617806891422e-04, -9.126962046755693e-04, 0.000000000000000e+00, -9.099237530941451e-04, -2.595116472005973e-02, 0.000000000000000e+00, -2.588908145308252e-02, -3.800106052616464e-02, 0.000000000000000e+00, -2.732905526419947e-249, -2.155109720776143e-08, 0.000000000000000e+00, 0.000000000000000e+00, -1.630869882961813e-225, 0.000000000000000e+00, -8.079594480619528e-229, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
