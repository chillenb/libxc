
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_mgga_xc_edmggah_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_edmggah", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-9.253858638348744e-01, -9.915354397200554e-01, -3.984250404002251e-01, -1.173065054595286e-01, -6.765469273554985e-02, -1.242503894762664e-01, -3.259401590620280e-02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_mgga_xc_edmggah_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_edmggah", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-1.610975209898260e+00, -1.613651143160787e+00, -1.431070217162960e+00, -1.432149210882682e+00, -4.183496010367748e-01, -4.178379827516767e-01, -1.829279048726331e-01, -1.220211652990912e-01, -6.013063038772694e-02, -3.627662941453050e-02, -6.064405349532165e-02, -6.046479538373314e-02, -2.334149974498871e-03, -9.496969488591391e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_xc_edmggah_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_edmggah", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [1.725199057919565e-05, 3.535197127625659e-05, 1.725191471222200e-05, -7.145669500900076e-04, 1.977440542960562e-04, -7.118056274227913e-04, -3.802784450879540e-02, 9.739044026124233e-02, -3.798210122502996e-02, -2.468806806369606e+00, 2.294242738482627e-03, -6.740799686866050e+02, -7.126107364946124e+01, 6.023271020286536e-06, -3.356252584746860e+07, -5.928963273837867e+02, 1.164855269195831e-02, -5.990016917494843e+02, -1.048465335968010e+08, 0.000000000000000e+00, -3.217053591117614e+08]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_xc_edmggah_Li_2_vlapl():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_edmggah", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vlapl"].flatten()
    ref_tgt = [-1.204588505709349e-04, -1.201645346868643e-04, -3.746744114654198e-03, -3.743201897423057e-03, -6.683647421432553e-03, -6.676109593333680e-03, -2.370367858516745e-02, -2.157794703383764e-03, -4.260479086436416e-02, -3.418641307918670e-03, -2.201333717915108e-03, -2.175828187670122e-03, -3.182513105129226e-03, -3.508454029363436e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_xc_edmggah_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_edmggah", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [-8.951873830065592e-04, -8.976053805254673e-04, 1.240924862090529e-02, 1.239288573873080e-02, 9.155820013258643e-03, 9.133144453853093e-03, 9.477067219535777e-02, 8.609146752862047e-03, 1.704191562552016e-01, 1.367456163054625e-02, 8.805206062461522e-03, 8.703184878288054e-03, 1.273005242051690e-02, 1.403381611745374e-02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
