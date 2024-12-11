
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_ms2bs_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ms2bs", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.826888021306606e+00, -1.214406813735091e+00, -2.873350529236384e-01, -1.692119854652218e-01, -6.193928718957031e-02, -1.852539395283411e-02, -3.185123251419169e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_ms2bs_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ms2bs", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.627134452292521e+00, -2.629865298173992e+00, -1.695538931689543e+00, -1.697735145864528e+00, -3.843956621132197e-01, -3.842822345310501e-01, -2.360238778361899e-01, -2.353561213485592e-02, -8.304678220926826e-02, -7.479200723728213e-04, -2.479733550720072e-02, -2.456629348724751e-02, -4.995701690963720e-04, -2.162505891704757e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_ms2bs_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ms2bs", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-8.019565829947531e-04, 0.000000000000000e+00, -7.994184458798814e-04, -1.655472721551292e-03, 0.000000000000000e+00, -1.658999072352430e-03, -2.139078073191839e-01, 0.000000000000000e+00, -2.144265014013001e-01, -1.070864903873337e+01, 0.000000000000000e+00, -2.991548054022322e-01, -1.031118371315641e+02, 0.000000000000000e+00, -1.916698100148620e+00, -1.276689819691941e-04, 0.000000000000000e+00, -2.838585809893152e-01, -8.748553208940801e-11, 0.000000000000000e+00, -2.025870855699590e+12]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_ms2bs_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ms2bs", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [2.745170323921371e-02, 2.744456533489537e-02, 1.127087799110529e-02, 1.137698094394206e-02, 2.203178198256410e-04, 2.307210867328488e-04, 2.941861535402159e-01, 2.098042097048271e-24, 3.860374147608373e-03, 1.967965606929963e-30, -1.784664603106567e-30, 8.425806116442476e-25, 4.467394546105110e-41, 3.375394739411110e-13]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
