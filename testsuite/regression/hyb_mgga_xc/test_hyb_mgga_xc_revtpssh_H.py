
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_mgga_xc_revtpssh_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_revtpssh", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-6.417960202054711e-01, -5.368321819459319e-01, -3.186431193267520e-01, -9.411806485513308e-02, -5.390070995725960e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_mgga_xc_revtpssh_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_revtpssh", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-8.093519048897925e-01, -6.417727865006351e-02, -7.142724016511001e-01, -2.505892441260956e-01, -4.214523277220083e-01, -1.951032357971834e-01, -1.230348415186342e-01, -9.049038861567087e-02, -7.095338365442299e-03, -7.028460031535879e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_xc_revtpssh_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_revtpssh", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-1.890445928264474e+00, 5.250111154972676e+00, 2.625055577486338e+00, -2.672108981823247e-03, 4.866817105678451e-02, 2.433408552839226e-02, 7.793625211324687e-02, 3.994554430141354e-01, 1.997277215070677e-01, 3.280749155676214e+01, 6.695547178407861e+01, 3.347773589203931e+01, 5.368726089270803e+06, 1.031701807315747e+07, 5.158509036578733e+06]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_xc_revtpssh_H_2_vtau():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_revtpssh", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [4.499403716837986e+00, -6.259673229373131e+00, 6.366313135533349e-03, -8.191707933254746e-69, 1.746630554038811e-03, -3.436759266207299e-66, -5.616482819597941e-04, -4.511491573768511e-55, -2.328221366196083e-08, -7.303672785988166e-40]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
