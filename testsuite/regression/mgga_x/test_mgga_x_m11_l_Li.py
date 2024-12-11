
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_m11_l_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_m11_l", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.694280430137003e+00, -1.422954762322920e+00, -1.397216289016358e-01, -2.707141615638871e-01, -1.352448772045962e-02, -8.377293958475121e-02, -1.288546651194454e-03])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_m11_l_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_m11_l", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-2.315703476657630e-01, -2.199498835780616e-01, -2.117095635659486e+00, -2.114924308036384e+00, -4.925776242711631e-01, -5.013556237757586e-01, -2.849085111140817e-01, -1.056359385056528e-01, -2.621988311184762e-02, -3.378753941008479e-03, -1.125171916175825e-01, -1.102296837216500e-01, -2.256880509756670e-03, -2.183767516652512e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_m11_l_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_m11_l", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-3.189082187649023e-03, 0.000000000000000e+00, -3.192491987739997e-03, -2.014213495095162e-03, 0.000000000000000e+00, -2.022245506165675e-03, -1.640582930383730e+00, 0.000000000000000e+00, -1.607012593473967e+00, 1.255378553395713e+02, 0.000000000000000e+00, -6.406418716089484e+00, -1.768424647233872e+03, 0.000000000000000e+00, -4.085260754813493e+01, -2.731878087848959e-03, 0.000000000000000e+00, -6.081205980762871e+00, -1.864661961262304e-09, 0.000000000000000e+00, -3.720925171068501e+13])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_m11_l_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_m11_l", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([-2.816941932354801e-01, -2.833585762522658e-01, 2.938995603705938e-02, 2.879706284638446e-02, 5.381486923691645e-02, 5.558127432466591e-02, -9.962941352493225e-01, -3.754835591890785e-05, 3.102509604253527e-02, -7.611298876859960e-09, -1.871689296088310e-08, -4.056558628383056e-05, -1.036690141416301e-19, -4.190951262207030e-10])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
