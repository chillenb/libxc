
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_c_rscan_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_rscan", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-2.565339835877350e-02, -2.438245086237930e-02, -1.478860048148416e-02, -6.628036638934437e-05, -1.371567820689784e-08, -9.702775457104781e-04, -2.343199001860853e-06])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_c_rscan_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_rscan", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-2.821765506005461e-02, -2.814486201849026e-02, -2.858545376836940e-02, -2.852009591872649e-02, -5.063509245842045e-02, -5.065509384093968e-02, -3.889562270268793e-04, -1.583327632021317e-01, -9.058263011202383e-09, -8.050532869593560e-02, -1.833586322610124e-03, -1.850665352756899e-03, -3.941189892834274e-06, -6.723003196124887e-06])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_rscan_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_rscan", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([1.789812516446464e-06, 3.579625032892927e-06, 1.789812516446464e-06, 9.571112072400234e-06, 1.914222414480047e-05, 9.571112072400234e-06, 1.934192194294116e-02, 3.868384388588231e-02, 1.934192194294116e-02, 5.322175212739141e-01, 1.064435042547828e+00, 5.322175212739141e-01, 1.477384132723691e-05, 2.954768265447383e-05, 1.477384132723691e-05, 2.761173024152648e+00, 5.522346048305296e+00, 2.761173024152648e+00, 2.619901438994025e+03, 5.239802877988051e+03, 2.619901438994025e+03])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_rscan_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_rscan", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([-2.105069673857442e-08, -2.105069673857441e-08, -8.458084594880520e-07, -8.458084594880518e-07, -8.240394607201149e-03, -8.240394607201145e-03, -2.030355780601376e-02, -2.030355780600927e-02, -1.195738410689188e-12, -1.195738409723143e-12, -1.311959429187563e-06, -1.311959429187564e-06, -2.017997649494923e-13, -2.017997649494924e-13])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
