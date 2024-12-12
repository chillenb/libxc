
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_revtm_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_revtm", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.941167231446866e+00, -1.359991641460525e+00, -3.810492999980560e-01, -1.749370562544685e-01, -7.696889665316035e-02, -4.837356165758987e-02, -4.060527276031686e-03])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_revtm_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_revtm", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-2.544040561770273e+00, -2.546812217908645e+00, -1.714085809353967e+00, -1.716251783316255e+00, -3.548719279810639e-01, -3.550844173568805e-01, -2.300913566697487e-01, -2.085620783475743e-02, -7.835920709729913e-02, -2.079140641185672e-03, -4.433055445677871e-02, -2.155148277148767e-02, -3.652337168665104e-03, -1.323262538611992e-03])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_revtm_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_revtm", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-8.460759220957375e-05, 0.000000000000000e+00, -8.228431286737432e-05, -7.550222382039738e-04, 0.000000000000000e+00, -7.431093488334087e-04, -1.186944910398680e-01, 0.000000000000000e+00, -1.193030552531490e-01, -3.234334722420400e+00, 0.000000000000000e+00, -5.047770576772503e+02, -8.230979447267057e+01, 0.000000000000000e+00, -4.240807151892634e+06, -2.402475042372733e+02, 0.000000000000000e+00, -4.456805425096432e+02, -5.301734654968568e+06, 0.000000000000000e+00, -2.576093846089367e+07])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_revtm_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_revtm", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([9.018192546225259e-04, 7.951732407771976e-04, 6.556961400345957e-03, 6.385493248826587e-03, 1.681013641245068e-02, 1.720746352005497e-02, 9.755927183529828e-02, 3.895036744553971e-03, 1.117684881360377e-01, 1.039469548541425e-03, -1.387451973542734e-06, 3.912939835595237e-03, -3.193842845719274e-14, 6.760051098531834e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
