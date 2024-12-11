
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_ktbm_7_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_7", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.926831329724589e+00, -1.279174869030424e+00, -2.767931060854950e-01, -1.773743088473230e-01, -5.794203987560835e-02, -1.245707754168807e-02, -2.295745915676966e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_ktbm_7_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_7", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-2.717887151574599e+00, -2.720508216226289e+00, -1.879700922031071e+00, -1.881403370904043e+00, -3.484409878426437e-01, -3.478889612193946e-01, -2.453090253517100e-01, -1.482519144106904e-02, -7.590820877445166e-02, -4.701911621557202e-04, -1.558984176640292e-02, -1.547630262055204e-02, -3.140606937526585e-04, -2.234355474028027e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_ktbm_7_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_7", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-5.347196011065136e-04, 0.000000000000000e+00, -5.328898581176149e-04, -2.052605410758392e-03, 0.000000000000000e+00, -2.046863236024777e-03, -5.514129591366273e-02, 0.000000000000000e+00, -5.730782187208285e-02, -8.305812720147040e+00, 0.000000000000000e+00, -2.633170778272561e+01, -7.563835094143124e+01, 0.000000000000000e+00, -6.600441390161775e+04, -4.900479970377544e-01, 0.000000000000000e+00, -2.354249109634549e+01, -9.994974559807334e-01, 0.000000000000000e+00, -1.390852413133610e+05])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_ktbm_7_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_7", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([2.138715469027415e-02, 2.137105130551771e-02, 2.558866449811645e-02, 2.559464887292037e-02, -3.796526013074207e-03, -3.891185101437635e-03, 2.485493657043367e-01, 3.364930783725260e-04, -1.128556859944306e-02, 2.689257490231666e-05, 7.277882790167296e-06, 3.422679885227018e-04, 1.213550374282676e-10, -1.098781453286224e-11])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
