
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_c_lypr_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_lypr", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-5.221201512135900e-02, -4.525526842353408e-02, 3.877743108779572e-03, -1.415546268912226e-05, -1.541576556398532e-09, -7.868583539840126e-09, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_c_lypr_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_lypr", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-6.186669997178539e-02, -6.169776841558702e-02, -6.187591581515675e-02, -6.171206253962629e-02, -6.060507613540040e-02, -6.073212926182168e-02, -5.407674375864676e-06, -4.323938714313268e-02, -1.296721942700672e-09, -9.048413404837391e-03, -6.869048425578699e-08, -6.903513961385191e-08, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_c_lypr_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_lypr", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [6.629315334083007e-06, 6.377456738086418e-06, 6.522683210503523e-06, 3.996479852628411e-05, 4.413025517428745e-05, 3.944225701890844e-05, 1.947237647048285e-02, 3.253606485805580e-02, 1.951794560708919e-02, 4.647457502198000e-05, 7.216492012456489e-02, 5.411653225596216e-02, 5.791626721194511e-14, 2.566359735238707e-08, 1.924766888949017e-08, 1.385076518380370e-153, 2.763910643695453e-153, 1.381037855692333e-153, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
