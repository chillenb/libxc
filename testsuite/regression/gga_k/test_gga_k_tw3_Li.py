
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_k_tw3_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_tw3", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([1.642826749493245e+01, 8.171876391340183e+00, 6.439074059280279e-01, 1.323443753920523e-01, 2.668793527533834e-02, 1.259482149377634e-03, 4.477941399684870e-07])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_k_tw3_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_tw3", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([2.592050926252993e+01, 2.596800314619294e+01, 1.232497653780632e+01, 1.234628647255587e+01, 8.273338720604700e-01, 8.273056887105872e-01, 2.135638109707055e-01, 1.910379042616568e-03, 3.382770078180022e-02, 1.924400152856325e-06, 2.112132396896842e-03, 2.081683014729214e-03, 8.585678343235938e-07, 4.339127767843812e-07])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_k_tw3_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_tw3", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([2.460839656596319e-03, 0.000000000000000e+00, 2.454575579231360e-03, 6.762873588357189e-03, 0.000000000000000e+00, 6.746919574551705e-03, 1.193290682003090e-01, 0.000000000000000e+00, 1.189824464674379e-01, 3.446409618414119e+00, 0.000000000000000e+00, 1.657622118207762e-02, 2.330949130984319e+01, 0.000000000000000e+00, 3.362795552425489e-03, 1.771366586780910e-02, 0.000000000000000e+00, 1.642111851727920e-02, 1.635117443972039e-03, 0.000000000000000e+00, 1.663883354464013e-03])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
