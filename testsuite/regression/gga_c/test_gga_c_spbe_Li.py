
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_c_spbe_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_spbe", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-6.175620746326106e-02, -4.675106914994890e-02, -8.669368341489098e-03, -1.537637609351862e-02, -3.084567310992860e-03, -6.804831282880217e-06, -1.735490764499022e-10]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_c_spbe_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_spbe", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-1.124489043156535e-01, -1.123220804778149e-01, -9.656160720511150e-02, -9.646487322411618e-02, -2.743170950348474e-02, -2.744350837439387e-02, -2.355705487466840e-02, -1.025774444907093e-01, -9.273039841565356e-03, 2.713174711093266e-01, -2.619514458202043e-05, -2.635389980588749e-05, -6.509530103601777e-10, -7.993047223514680e-10]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_c_spbe_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_spbe", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [4.118359626137914e-05, 8.236719252275828e-05, 4.118359626137914e-05, 1.264996248508756e-04, 2.529992497017513e-04, 1.264996248508756e-04, 4.488798692606220e-03, 8.977597385212442e-03, 4.488798692606220e-03, 2.925535698780909e+00, 5.851071397561817e+00, 2.925535698780909e+00, 1.282613839528337e+01, 2.565227679056673e+01, 1.282613839528337e+01, 7.501860813234250e-02, 1.500372162646836e-01, 7.501860813234250e-02, 7.761352151467689e-01, 1.552270430408909e+00, 7.761352151467689e-01]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
