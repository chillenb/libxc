
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_xc_hcth_407p_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_xc_hcth_407p", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.915072373648699e+00, -1.343656576222049e+00, -4.819318598899766e-01, -1.764544741502675e-01, -9.126799346351594e-02, 8.171636776785580e-03, 5.467122379015572e-05])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_xc_hcth_407p_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_xc_hcth_407p", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-2.567709455469324e+00, -2.570026613372628e+00, -1.736800353040209e+00, -1.738295461566880e+00, -3.705213609637504e-01, -3.716455478640301e-01, -2.478849180483139e-01, 1.703467798196031e+00, -5.629509427786665e-02, 1.076114252930054e+00, 7.181721189000569e-03, 9.294855210936221e-03, -5.591844127703577e-04, 1.832214066275483e-03])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_xc_hcth_407p_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_xc_hcth_407p", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([5.079560684848200e-05, 0.000000000000000e+00, 5.099950315480741e-05, -2.105549669018218e-04, 0.000000000000000e+00, -2.089058089674884e-04, -1.303702909252023e-01, 0.000000000000000e+00, -1.299065984471741e-01, 6.502240919225048e+00, 0.000000000000000e+00, 4.211714957286130e+02, -1.391691656345442e+02, 0.000000000000000e+00, 4.998904024699638e+04, 3.851323686413218e+00, 0.000000000000000e+00, 4.246067614600889e+00, 2.655768536026923e+00, 0.000000000000000e+00, 1.237504328366049e+02])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
