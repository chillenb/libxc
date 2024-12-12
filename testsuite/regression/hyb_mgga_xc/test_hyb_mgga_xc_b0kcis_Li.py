
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_mgga_xc_b0kcis_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_b0kcis", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.434294103644828e+00, -1.036853686646574e+00, -3.454924649522592e-01, -1.219542076477373e-01, -6.094684496633276e-02, -9.999133362320536e-02, -4.021055101377800e-02])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_mgga_xc_b0kcis_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_b0kcis", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-1.850761214489899e+00, -1.851991397345073e+00, -1.290405852303311e+00, -1.291094155338896e+00, -3.141291073755342e-01, -3.137017040284683e-01, -1.851193293552242e-01, -2.427606455416831e-01, -6.338110618580302e-02, -9.500907651421309e-02, -2.784391329404227e-02, -2.793728181360515e-02, -5.596121978651536e-03, -4.840079364276042e-03])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_xc_b0kcis_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_b0kcis", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([2.598892569087200e-05, 1.229587511315676e-04, 2.648032614531168e-05, -2.399630700979413e-04, 4.052085216529476e-04, -2.378454538963650e-04, -7.335866283022092e-02, 2.085622589088825e-02, -7.328471699160909e-02, 3.470423288062436e+01, 9.725974969355788e+00, -9.998282317832469e+02, -8.598874976642843e+00, 5.183795594681042e+01, -3.637724883489625e+07, -8.726080929941546e+02, 2.084129777272501e+00, -8.740012228832335e+02, -1.079999033921980e+08, 5.287011995525877e+02, -3.217205774228383e+08])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_xc_b0kcis_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_b0kcis", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([-1.291234116099550e-02, -1.293488991512605e-02, -9.755273040835708e-03, -9.781755149360607e-03, -2.094644272068747e-03, -2.202983213579078e-03, -1.380984854416353e+00, -4.887621591907894e-06, -1.168176537530762e-01, -3.145742044544087e-08, -2.296813269428660e-09, -5.054936212203982e-06, -6.406366243580442e-19, -7.391544129278327e-09])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
