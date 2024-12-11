
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_k_dk_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_dk", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([1.638131020467760e+01, 8.168390853837726e+00, 5.917870546957691e-01, 1.330024220758787e-01, 2.281197274897362e-02, 3.051984165595038e+00, 1.353596813377780e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_k_dk_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_dk", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([2.581580315367221e+01, 2.586354023447704e+01, 1.217717433097929e+01, 1.219833073228922e+01, -3.386142498172530e-01, -3.518344678218823e-01, 2.134030520684286e-01, -3.087244405671780e+00, 4.364870871253375e-02, -1.207330039309344e+00, -3.066285537558947e+00, -3.171270281872380e+00, -1.415023323617867e+00, -1.182719219857836e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_k_dk_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_dk", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([2.506300734640962e-03, 0.000000000000000e+00, 2.499207182170384e-03, 7.511239049530530e-03, 0.000000000000000e+00, 7.493190290667064e-03, 6.421036740480427e-01, 0.000000000000000e+00, 6.485493428463730e-01, 3.242082030259221e+00, 0.000000000000000e+00, 7.810445193142218e+04, -1.231776049790627e+01, 0.000000000000000e+00, 2.448422813091073e+09, 6.716742797685522e+04, 0.000000000000000e+00, 6.865492579072007e+04, 8.216163589204875e+09, 0.000000000000000e+00, 2.286797491532547e+10])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
