
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_mgga_x_m06_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_m06", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.265379438874258e+00, -8.681055447171724e-01, -8.195227072858052e-02, -1.117871710796750e-01, -3.817518641446595e-02, -5.239426501254602e-02, -8.915857721243011e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_mgga_x_m06_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_m06", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.099314937560101e+00, -2.100705436905488e+00, -8.996696234632432e-01, -8.995952474056021e-01, -3.353537906657859e-01, -3.416333000460681e-01, -7.469590742397332e-02, -6.611633497839299e-02, -8.342387889480815e-02, -2.119118005170103e-03, -7.024729318210184e-02, -6.897516458302320e-02, -1.415487062164273e-03, -5.577748666793630e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_x_m06_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_m06", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-1.371600712736231e-04, 0.000000000000000e+00, -1.367235878546195e-04, -6.809925322720394e-04, 0.000000000000000e+00, -6.778082676741958e-04, -6.242657959052218e-01, 0.000000000000000e+00, -6.183363548714854e-01, -1.846972034904183e+00, 0.000000000000000e+00, -2.322943753020586e+00, -2.135571901777646e+02, 0.000000000000000e+00, -1.491953909344895e+01, -9.937234980696432e-04, 0.000000000000000e+00, -2.203836312383504e+00, -6.809898400214372e-10, 0.000000000000000e+00, -8.945432744539867e+12]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_x_m06_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_m06", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [5.877965856202234e-02, 5.867485224359351e-02, -3.765430243091879e-02, -3.776384362510755e-02, 4.147309367462244e-02, 4.287811000010288e-02, -2.116099068704159e+00, -3.692363413690218e-05, 2.723192739954851e-01, -7.551211553621607e-09, -1.833350038384630e-08, -3.985627257994662e-05, -1.028889863300003e-19, -8.745936467888272e-09]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
