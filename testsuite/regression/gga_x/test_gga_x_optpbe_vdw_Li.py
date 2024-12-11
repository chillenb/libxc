
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_x_optpbe_vdw_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_optpbe_vdw", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.783513747657432e+00, -1.270026134597423e+00, -4.192699031596017e-01, -1.594251874027100e-01, -7.937511901219664e-02, -2.331289208945716e-02, -4.357858478419545e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_x_optpbe_vdw_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_optpbe_vdw", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-2.253829974354882e+00, -2.255961657412015e+00, -1.528819231867551e+00, -1.530197477659587e+00, -3.567937155740666e-01, -3.569032229171011e-01, -2.060051390555475e-01, -2.961080834864206e-02, -7.175991846562026e-02, -9.418720536473126e-04, -3.112730157214300e-02, -3.090572598355331e-02, -6.291193376019066e-04, -4.472468733607889e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_x_optpbe_vdw_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_optpbe_vdw", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-2.110681268892109e-04, 0.000000000000000e+00, -2.103285313417873e-04, -8.633504367505192e-04, 0.000000000000000e+00, -8.605176993920386e-04, -9.805950415453507e-02, 0.000000000000000e+00, -9.791184347062572e-02, -3.222450902742892e+00, 0.000000000000000e+00, -5.570072126008023e-01, -7.456095112179133e+01, 0.000000000000000e+00, -3.568539448887300e+00, -5.659364278002109e-01, 0.000000000000000e+00, -5.285287236697586e-01, -2.597771561639526e+00, 0.000000000000000e+00, -3.718444949053000e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
