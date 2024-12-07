
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_gga_xc_b3pw91_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_b3pw91", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.505330240817907e+00, -1.079874436783751e+00, -3.469930003365191e-01, -1.439411847549322e-01, -6.674468972817341e-02, -9.810764689672513e-02, -3.865105003265951e-02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_gga_xc_b3pw91_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_b3pw91", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-1.910402199363436e+00, -1.911961121467742e+00, -1.324355991846859e+00, -1.325339408472349e+00, -3.022893855040826e-01, -3.021998536311291e-01, -1.878748424992709e-01, -1.291612041923384e-01, -6.757734817302391e-02, 3.092551814452978e-01, -2.923809528391998e-02, -2.937415858678322e-02, -5.434254539613765e-03, -4.718858178326677e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_gga_xc_b3pw91_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_b3pw91", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-1.649896724348123e-04, 6.453534174202383e-05, -1.643372276995607e-04, -6.360448127164752e-04, 2.118487154551634e-04, -6.337215611245657e-04, -7.950132942491484e-02, 6.140536786824862e-03, -7.947703169534509e-02, -5.333894682033008e-01, 5.282761836866798e+00, -9.618632741481416e+02, -4.557310564576305e+01, 1.915084822813345e+01, -3.492217420093289e+07, -8.387040342069930e+02, 2.805909535933161e-04, -8.400424267501455e+02, -1.036801610330846e+08, 2.603264411672309e-06, -3.088520081156201e+08]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
