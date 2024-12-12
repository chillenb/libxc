
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_c_revtm_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_revtm", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-3.456882377531017e-02, -2.783336604550721e-02, -3.072578253993606e-03, -2.533635879603488e-03, -8.138077589633276e-09, -2.002734956871438e-08, -5.655232412814755e-16])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_c_revtm_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_revtm", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-1.004902892177829e-01, -1.003889200553392e-01, -8.291455986541144e-02, -8.283028871972657e-02, -1.609826579026751e-02, -1.612135950354526e-02, -2.978309410772609e-02, -1.577396873720388e-01, -4.144721674599425e-03, -7.088312715036007e-03, -1.285668341159558e-07, -1.292209447992731e-07, -3.577733455464826e-15, -4.233361194955901e-15])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_revtm_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_revtm", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([1.009910344508579e-04, 2.835724043648179e-04, 1.010986497872385e-04, 2.146238375786829e-04, 6.726760871730478e-04, 2.149593611147622e-04, 3.057941566962698e-03, 8.380954067275657e-03, 3.066831249972213e-03, 3.383507186677897e+01, 7.392617200503059e+01, 1.137377844710411e+02, 2.418546794759335e+01, 4.836762938403192e+01, 9.728105195538441e+03, 4.416235898119373e-04, 8.836221003985414e-04, 4.416474641964310e-04, 5.059265615691109e-06, 1.011868011772730e-05, 5.059265615716128e-06])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_revtm_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_revtm", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([-1.101448855623617e-02, -1.101448855623617e-02, -7.408592660852545e-03, -7.408592660852542e-03, -3.267871726610102e-04, -3.267871726610099e-04, -1.172452845215676e+00, -1.172452845215417e+00, -5.783872890105329e-02, -5.783872885432500e-02, -1.231150597339579e-13, -1.231150597339579e-13, -2.845928495864769e-31, -2.845928495864771e-31])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
