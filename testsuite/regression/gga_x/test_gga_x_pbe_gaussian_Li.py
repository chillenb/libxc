
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_x_pbe_gaussian_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_pbe_gaussian", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.794388704619162e+00, -1.283597959297820e+00, -4.160513789629903e-01, -1.600245076108190e-01, -8.050301599932122e-02, -2.054449214044466e-02, -3.838587880471190e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_x_pbe_gaussian_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_pbe_gaussian", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.242612748545779e+00, -2.244746487559736e+00, -1.518992516009380e+00, -1.520361569823459e+00, -4.007631837994660e-01, -4.009399216330560e-01, -2.053103199865797e-01, -2.611574043371735e-02, -7.640084781769188e-02, -8.296435153811291e-04, -2.745724681794555e-02, -2.725995225461296e-02, -5.541556588241993e-04, -3.939542941250926e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_x_pbe_gaussian_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_pbe_gaussian", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-2.551460062525215e-04, 0.000000000000000e+00, -2.542671841365495e-04, -1.010457651106938e-03, 0.000000000000000e+00, -1.007215478545735e-03, -7.467192054868257e-02, 0.000000000000000e+00, -7.448762702206815e-02, -3.950061551428108e+00, 0.000000000000000e+00, -2.777227733618850e-01, -6.769631939990828e+01, 0.000000000000000e+00, -1.776492849752950e+00, -2.822254913201755e-01, 0.000000000000000e+00, -2.635488361172218e-01, -1.293221299768140e+00, 0.000000000000000e+00, -1.851113238663231e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
