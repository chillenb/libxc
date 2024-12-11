
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_gga_xc_hflyp_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_hflyp", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-5.538903098368998e-02, -4.949790488041998e-02, 1.413408320172361e-02, -1.276112437397293e-05, -2.334131483445219e-09, -2.078188623821984e-03, -3.003770849422545e-05])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_gga_xc_hflyp_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_hflyp", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-6.387961131095844e-02, -6.371176627512085e-02, -6.461227879900613e-02, -6.444379768591930e-02, -1.043591405809932e-01, -1.047837406785314e-01, -2.225850205384591e-05, -8.222888742779609e-02, -2.804578703192513e-09, -3.029946428134138e-02, -2.687423197831241e-03, -2.778500684814050e-03, -2.079686215637139e-05, -9.360839371319271e-05])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_gga_xc_hflyp_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_hflyp", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([6.601419480515535e-06, 5.222815421851711e-06, 6.477346546289099e-06, 4.053330432132051e-05, 3.646941789248587e-05, 3.987902204897787e-05, 3.461076381698536e-02, 4.773762863586187e-02, 3.476915844521113e-02, -4.194706172367205e-04, 4.596134769453040e+00, 3.448547549012627e+00, 2.178425425261535e-06, 2.356939734329661e+01, 1.767704995343962e+01, 4.086933530334335e-02, 7.936097321777658e-02, 4.107927410229123e-02, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
