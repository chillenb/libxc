
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_k_csk4_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_k_csk4", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [1.636036084321611e+01, 8.103293700138579e+00, 3.215798213061692e+00, 1.328677494702689e-01, "nan", 1.176944135381742e+166, 8.879178266767201e+257]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_k_csk4_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_k_csk4", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [2.595105458303196e+01, 2.599869596156329e+01, 1.227997133723978e+01, 1.230141899237861e+01, -3.210589206152795e+00, -3.220182520439590e+00, 2.138761485345217e-01, -3.057055387610054e+00, "nan", "nan", -6.930966674342957e+150, 5.674799568746964e+150, 6.452647315997695e+242, 2.309007955429813e+243]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_k_csk4_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_k_csk4", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [2.141325803425453e-03, 0.000000000000000e+00, 2.135548041553573e-03, 6.398161520751163e-03, 0.000000000000000e+00, 6.381851718619596e-03, 4.153244234607504e+00, 0.000000000000000e+00, 4.158550604203024e+00, 2.894480549942090e+00, 0.000000000000000e+00, 7.829799911346240e+04, 4.175310184808651e+02, 0.000000000000000e+00, "nan", 6.733488483171388e+04, 0.000000000000000e+00, 7.647285960080892e+03, 8.236143114997762e+09, 0.000000000000000e+00, 2.547064861199871e+09]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_k_csk4_Li_2_vlapl():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_k_csk4", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vlapl"].flatten()
    ref_tgt = [1.666666666666667e-01, 1.666666666666666e-01, 1.666666537462715e-01, 1.666666548150407e-01, 7.335869500059731e-06, 7.196670299775354e-06, 1.666666666666666e-01, 2.792314828664535e-07, 2.783508692741352e-04, "nan", -3.822606223613729e-07, 1.666666666666668e-01, 0.000000000000000e+00, 1.666666666666666e-01]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_k_csk4_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_k_csk4", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
