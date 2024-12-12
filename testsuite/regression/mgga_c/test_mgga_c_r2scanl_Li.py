
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_c_r2scanl_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_r2scanl", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-2.565339835151432e-02, -2.438245026053473e-02, -1.656502786924554e-02, -1.254624297729948e-04, -3.585436227863253e-08, -1.059122968721063e-03, -4.702749938105233e-06])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_c_r2scanl_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_r2scanl", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-2.821749363228022e-02, -2.814484573646542e-02, -2.858443147364482e-02, -2.851419297144210e-02, -7.636747551876696e-02, -7.443786116337577e-02, -4.510401600103850e-06, 1.138670459391505e-01, -2.689962519193174e-08, 1.323869249931693e-01, -2.001588623452305e-03, -1.970377247603045e-03, -9.071653506893739e-07, -3.275099383137771e-05])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_r2scanl_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_r2scanl", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-1.334215055360877e-05, 3.376108331953670e-05, -1.326060316906006e-05, -5.534735971688365e-05, 1.485510331169195e-04, -5.501693963932295e-05, 3.402088944668620e-02, 5.644439419130742e-02, 3.162149824322134e-02, 1.159458673756882e-02, 4.181094236312343e+00, -7.595828367502868e+03, 3.498870890787982e-05, 1.567411078430473e+02, -4.592727274946454e+08, 4.598404864347916e-01, 1.137061882735642e+01, -1.088419816223767e+00, -2.654881296784491e+05, 1.526008745697776e+06, -2.099135049821117e+06])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_r2scanl_Li_2_vlapl():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_r2scanl", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vlapl"].flatten()
    ref_tgt = numpy.asarray([0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, -2.583541735966628e-03, -2.471702821825180e-03, 0.000000000000000e+00, 2.405603778310393e-03, 0.000000000000000e+00, -8.145891788906022e-05, 1.477273108744791e-10, 3.005298910049602e-06, 3.643176840589023e-20, -6.426826995251549e-09])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
