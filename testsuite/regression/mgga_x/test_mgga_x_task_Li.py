
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_task_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_task", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.963565655996748e+00, -1.318707048753535e+00, -1.641777123396462e-01, -1.789559392101754e-01, -4.269262876116411e-02, -4.816589056338742e-03, -2.193952037138248e-05]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_task_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_task", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.691251850761223e+00, -2.693679451610557e+00, -1.946519976631435e+00, -1.947748467500788e+00, -2.699562915451559e-01, -2.722754825077104e-01, -2.429075454762884e-01, -1.124692921511025e-02, -7.917671678394259e-02, -1.105765744337069e-04, -5.911150703399405e-03, -1.184713620106153e-02, -3.384378308926089e-06, -1.043850862947584e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_task_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_task", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-1.942168283631652e-04, 0.000000000000000e+00, -1.934201241274763e-04, -1.512272716426158e-03, 0.000000000000000e+00, -1.504565776963170e-03, 1.441475698222872e-01, 0.000000000000000e+00, 1.415405704219290e-01, -3.013369896844550e+00, 0.000000000000000e+00, 2.805042303557318e+01, -9.135169587079316e+00, 0.000000000000000e+00, 2.662357818366936e+04, 3.064038849536413e-01, 0.000000000000000e+00, 2.519790168521046e+01, 1.940994653209789e-02, 0.000000000000000e+00, 2.043540934786205e+12]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_task_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_task", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [1.059838531595781e-02, 1.058279790095805e-02, 2.765665349912469e-02, 2.758432332360498e-02, 9.358440631672909e-03, 9.857213699892930e-03, 1.201421826032748e-01, 1.101738966237264e-12, 1.863253677500858e-01, 1.276860945545905e-16, 5.466011034843418e-16, 1.171820848926183e-12, 1.693561776168035e-33, 6.397358457301383e-12]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
