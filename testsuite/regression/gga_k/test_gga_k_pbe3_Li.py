
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_k_pbe3_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_pbe3", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [1.752410045623415e+01, 1.102059112870980e+01, 1.226055142035144e+00, 1.264079749699655e-01, 5.203273092676965e-02, 2.073254952325628e-03, 7.367767288229117e-07]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_k_pbe3_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_pbe3", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [1.468839492428347e+01, 1.472979249772926e+01, 6.122669492855392e+00, 6.130523968755480e+00, 1.771397233503108e+00, 1.771409836381236e+00, 1.603906088641316e-01, 3.146750983756929e-03, 6.482620174326775e-02, 3.166313719328210e-06, 3.479522455482699e-03, 3.429156631077724e-03, 1.412642957072516e-06, 7.139372582508250e-07]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_k_pbe3_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_pbe3", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [2.484785230699154e-02, 0.000000000000000e+00, 2.477227794676755e-02, 6.445174334827664e-02, 0.000000000000000e+00, 6.432923424416294e-02, 1.322142946329520e-01, 0.000000000000000e+00, 1.314566156437903e-01, 2.450348084933319e+01, 0.000000000000000e+00, 6.505780506779524e-03, 4.791142600981791e+01, 0.000000000000000e+00, 1.316916145723834e-03, 6.953962916909670e-03, 0.000000000000000e+00, 6.445783185306047e-03, 6.403318487353396e-04, 0.000000000000000e+00, 6.515963841077525e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
