
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_x_pbe_jsjr_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_pbe_jsjr", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.776826196037664e+00, -1.260686384542044e+00, -3.944813803006374e-01, -1.590720104573368e-01, -7.643845142337423e-02, -2.053891971465549e-02, -3.838586126492227e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_x_pbe_jsjr_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_pbe_jsjr", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-2.262715362974985e+00, -2.264838987681640e+00, -1.541118208815541e+00, -1.542492889121314e+00, -3.722730206853009e-01, -3.723522184562905e-01, -2.064742185722377e-01, -2.609624498990020e-02, -7.477745997663755e-02, -8.296417419761001e-04, -2.743440490209922e-02, -2.723835249403056e-02, -5.541551272887084e-04, -3.939540290510505e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_x_pbe_jsjr_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_pbe_jsjr", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-1.805553555844001e-04, 0.000000000000000e+00, -1.799251978634974e-04, -7.329387156210476e-04, 0.000000000000000e+00, -7.305464849331476e-04, -7.453817094626661e-02, 0.000000000000000e+00, -7.440705659133859e-02, -2.765568030623845e+00, 0.000000000000000e+00, -4.023868458054178e-01, -5.938971645194748e+01, 0.000000000000000e+00, -2.576797160936878e+00, -4.088582945584944e-01, 0.000000000000000e+00, -3.818242211400739e-01, -1.875816639556286e+00, 0.000000000000000e+00, -2.685039677764230e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
