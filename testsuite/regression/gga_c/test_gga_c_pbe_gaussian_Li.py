
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_c_pbe_gaussian_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_pbe_gaussian", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-5.998829931219104e-02, -4.345986130656022e-02, -3.078526374955454e-03, -1.516237945277877e-02, -1.348091239621072e-03, -7.609296216563866e-09, -1.790559887859811e-16])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_c_pbe_gaussian_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_pbe_gaussian", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-1.160121339009665e-01, -1.158863065743723e-01, -1.014559292869165e-01, -1.013614401806737e-01, -1.592565796034825e-02, -1.593131569313088e-02, -2.421901562877560e-02, -9.763981221025826e-02, -6.581792923311611e-03, 3.428094083148455e-01, -4.924674442190514e-08, -4.949413081365890e-08, -1.135844077213705e-15, -1.343902474113673e-15])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_c_pbe_gaussian_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_pbe_gaussian", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([4.595522569478208e-05, 9.191045138956413e-05, 4.595522569478208e-05, 1.490502773723655e-04, 2.981005547447310e-04, 1.490502773723655e-04, 3.124864518433215e-03, 6.249729036866429e-03, 3.124864518433215e-03, 3.381205134459136e+00, 6.762410268918273e+00, 3.381205134459136e+00, 1.129316781828352e+01, 2.258633563656705e+01, 1.129316781828352e+01, 1.678498292813975e-04, 3.356996585412902e-04, 1.678498292813975e-04, 1.606574004317374e-06, 3.212914604734337e-06, 1.606574004317374e-06])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
