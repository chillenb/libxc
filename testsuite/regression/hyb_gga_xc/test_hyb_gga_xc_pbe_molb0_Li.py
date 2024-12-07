
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_gga_xc_pbe_molb0_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_pbe_molb0", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.417591146176008e+00, -1.020972613382489e+00, -3.251589111374465e-01, -1.359149929704032e-01, -6.388497728340911e-02, -1.541026830807861e-02, -2.878940523658045e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_gga_xc_pbe_molb0_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_pbe_molb0", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-1.786797004532160e+00, -1.788273849528712e+00, -1.229959773898385e+00, -1.230884528509984e+00, -3.345708909936326e-01, -3.347417723716798e-01, -1.773832595739478e-01, -1.188024607807746e-01, -6.590121399157957e-02, 3.627635401251492e-01, -2.060075985397361e-02, -2.045236585090969e-02, -4.156167829038736e-04, -2.954657098238931e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_gga_xc_pbe_molb0_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_pbe_molb0", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-1.902670928413742e-04, 9.023626760506512e-05, -1.894650168219964e-04, -7.663946433879796e-04, 2.952245611818778e-04, -7.635018070827002e-04, -5.081436823450516e-02, 6.753471018556743e-03, -5.065749435035540e-02, -4.503218757869383e-01, 6.451036131242160e+00, 3.059671752189072e+00, -4.129758263520399e+01, 2.406024662208501e+01, 1.096980047253155e+01, -1.683559926354797e-01, 3.779858727432186e-04, -1.571980583200990e-01, -7.718735092934216e-01, 3.617816558929657e-06, -1.104858080946976e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
