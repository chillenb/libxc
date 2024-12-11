
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_mgga_xc_b0kcis_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_b0kcis", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.488996949000196e+00, -1.071452031254633e+00, -3.162327380179664e-01, -6.490633722514921e+79, -8.678955028441579e+80, -2.787774984917546e+109, -1.738494685683479e+56]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_mgga_xc_b0kcis_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_b0kcis", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-1.940647122473825e+00, -1.941917864884602e+00, -1.389976018455838e+00, -1.390653095966552e+00, -4.027272963656401e-01, -4.027055031834897e-01, 3.611447640826115e+66, -1.273466710330112e+81, 1.394701878022423e+71, -3.115562738843564e+85, -1.388867766145875e+107, -1.419304931011427e+107, -2.160109903279821e+54, -6.010828426846824e+40]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_xc_b0kcis_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_b0kcis", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [5.121185830499444e-04, 7.158614620626534e-04, 5.141125272759093e-04, 1.607448155167564e-03, 3.169722484299707e-03, 1.609555806894310e-03, 2.418953137257263e-01, 1.036350044974578e+00, 2.429262567932403e-01, 5.310244109139180e+01, 2.480702586574497e+01, 1.240464269262993e+01, 4.676807557355274e+02, 9.947538485610339e+02, 4.973900885944859e+02, 1.681218532513809e-03, 3.362416960914411e-03, 2.734109133888459e-03, 1.291118582227598e-08, 2.582237163654554e-08, -3.282012849162253e+12]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_xc_b0kcis_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_b0kcis", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [-3.035840979248775e-05, -7.034103415349772e-43, -5.956394438999101e-42, -5.937080900939433e-42, -2.811532432615062e-38, -2.963870361282260e-38, -2.045216027895315e-32, -4.887621591907894e-06, -2.771312176320362e-31, -3.145742044558463e-08, -2.303942272539986e-09, -5.054936212203982e-06, -6.406980012320759e-19, -1.508226637418507e-20]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
