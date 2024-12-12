
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_ms0_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ms0", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.995217476937865e+00, -1.390732561788530e+00, -3.510359427106702e-01, -1.800028930077205e-01, -7.401940112031487e-02, -1.469689152511463e-02, -2.744888499561409e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_ms0_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ms0", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-2.637561736967050e+00, -2.640003932478954e+00, -1.825040174201198e+00, -1.826932722279577e+00, -3.710443095901061e-01, -4.405931230033332e-01, -2.387711861456936e-01, -1.869577324389204e-02, -9.351901268795355e-02, -5.932608043378391e-04, -1.965863384310684e-02, -1.951624476315872e-02, -3.962646393581161e-04, -2.817079714720981e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_ms0_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ms0", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-3.704733713265761e-05, 0.000000000000000e+00, -3.669700455977528e-05, -1.529597665597859e-04, 0.000000000000000e+00, -1.478840872966519e-04, -1.038082984613669e-01, 0.000000000000000e+00, -1.358235520910183e-02, -7.683905477408067e-01, 0.000000000000000e+00, -6.420005604894172e-02, -1.132107992165222e+01, 0.000000000000000e+00, -9.170322997575647e-01, -6.546579001721009e-02, 0.000000000000000e+00, -6.092093809522077e-02, -2.991555973437640e-01, 0.000000000000000e+00, -1.217997901664907e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_ms0_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ms0", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([1.153566075011718e-05, 1.118288293653009e-11, 7.976907243514445e-05, 1.593034025786723e-17, 2.136392885661028e-02, 6.626866555597765e-11, 7.762715742432717e-03, 6.370438933669128e-18, 6.596551532443045e-07, 2.061970171548608e-10, 2.406302604472056e-21, 2.270021881361352e-18, -8.846871784847806e-38, 3.445308222741875e-11])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
