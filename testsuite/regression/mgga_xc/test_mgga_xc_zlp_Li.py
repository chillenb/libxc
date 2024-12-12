
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_xc_zlp_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_xc_zlp", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-3.761586782181289e+00, -1.418277190086019e+00, -3.882059339518433e-01, -1.532218880112836e-01, -6.826997238473907e-02, 5.067854864372802e+01, 5.177549717751713e+06])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_xc_zlp_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_xc_zlp", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-1.843809279810509e+00, -1.843809279810508e+00, -1.677003066540948e+00, -1.677003066540948e+00, -2.319953697156254e-01, -2.319953697156254e-01, -1.784933157147327e-01, -1.784933157146562e-01, -4.635834995338483e-02, -4.635834980981741e-02, -1.261240504248148e+01, -1.261240504248147e+01, -1.725756421855300e+06, -1.725756421855300e+06])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_xc_zlp_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_xc_zlp", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-8.595358253157347e-05, -1.719071650631469e-04, -8.595358253157347e-05, -3.724675735244926e-04, -7.449351470489852e-04, -3.724675735244926e-04, -1.136423469880652e-01, -2.272846939761305e-01, -1.136423469880652e-01, -3.317848281575371e+00, -6.635696563150742e+00, -3.317848281575371e+00, -1.346102288505678e+02, -2.692204577011357e+02, -1.346102288505678e+02, -4.752672821488794e+04, -9.505345642977588e+04, -4.752672821488794e+04, -4.761117573080667e+11, -9.522235146161334e+11, -4.761117573080667e+11])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_xc_zlp_Li_2_vlapl():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_xc_zlp", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vlapl"].flatten()
    ref_tgt = numpy.asarray([1.116518260168652e-03, 1.116518260168652e-03, 1.619144564985010e-03, 1.619144564985010e-03, 6.835947677298155e-03, 6.835947677298150e-03, 1.592567539510538e-02, 1.592567539510185e-02, 4.023965041066737e-02, 4.023965037815749e-02, 1.745458077097484e-01, 1.745458077097484e-01, 9.822140971797491e+00, 9.822140971797495e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
