
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_k_ge2_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_ge2", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [1.633469964697839e+01, 8.103293696691626e+00, 7.981069000130825e-01, 1.319641626028203e-01, 2.833910189186079e-02, 3.434026647886767e-01, 1.507666172918593e-01]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_k_ge2_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_ge2", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [2.595105458303197e+01, 2.599869596156329e+01, 1.227997208441638e+01, 1.230141967813711e+01, 3.782028747687335e-01, 3.765126437305246e-01, 2.138761485345217e-01, -3.386363269644200e-01, 2.599946375417305e-02, -1.344686676527339e-01, -3.357579634641953e-01, -3.475308790934636e-01, -1.576047663465392e-01, -1.317316077434998e-01]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_k_ge2_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_ge2", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [2.141325803425451e-03, 0.000000000000000e+00, 2.135548041553571e-03, 6.398157552748461e-03, 0.000000000000000e+00, 6.381848088126381e-03, 4.614896372377461e-01, 0.000000000000000e+00, 4.620789138692904e-01, 2.894480549942088e+00, 0.000000000000000e+00, 8.699790635302610e+03, 4.646130896403079e+01, 0.000000000000000e+00, 2.727085252418317e+08, 7.481638617199315e+03, 0.000000000000000e+00, 7.647285960080887e+03, 9.151270127775291e+08, 0.000000000000000e+00, 2.547064861199870e+09]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
